// Copyright (c) 2025, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import {execSync} from 'node:child_process';
import * as fs from 'node:fs';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import {Readable, Writable} from 'node:stream';
import {pipeline} from 'node:stream/promises';

import {describe, expect, it} from 'vitest';

import {
    ByteLimitExceededError,
    ByteLimitTransform,
    HeaptrackWrapper,
} from '../../lib/runtime-tools/heaptrack-wrapper.js';
import {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';

function collector() {
    const chunks: Buffer[] = [];
    const sink = new Writable({
        write(chunk: Buffer, _encoding, callback) {
            chunks.push(chunk);
            callback();
        },
    });
    return {sink, received: () => Buffer.concat(chunks)};
}

function chunksOf(...sizes: number[]) {
    return sizes.map((size, i) => Buffer.alloc(size, i + 1));
}

describe('ByteLimitTransform', () => {
    it('passes everything through when under the limit', async () => {
        const {sink, received} = collector();
        const limiter = new ByteLimitTransform(100);
        await pipeline(Readable.from(chunksOf(10, 20, 30)), limiter, sink);
        expect(received()).toEqual(Buffer.concat(chunksOf(10, 20, 30)));
        expect(limiter.bytesSeen).toEqual(60);
    });

    it('allows exactly the limit', async () => {
        const {sink, received} = collector();
        await pipeline(Readable.from(chunksOf(40, 60)), new ByteLimitTransform(100), sink);
        expect(received().length).toEqual(100);
    });

    it('fails the pipeline once the limit is exceeded, without passing on the overflowing chunk', async () => {
        const {sink, received} = collector();
        const source = Readable.from(chunksOf(40, 60, 1, 1000));
        const result = pipeline(source, new ByteLimitTransform(100), sink);
        await expect(result).rejects.toBeInstanceOf(ByteLimitExceededError);
        await expect(result).rejects.toThrow('100 bytes');
        expect(received()).toEqual(Buffer.concat(chunksOf(40, 60)));
        expect(source.destroyed).toBe(true);
        expect(sink.destroyed).toBe(true);
    });

    it('fails on a single chunk larger than the limit', async () => {
        const {sink, received} = collector();
        await expect(pipeline(Readable.from(chunksOf(101)), new ByteLimitTransform(100), sink)).rejects.toBeInstanceOf(
            ByteLimitExceededError,
        );
        expect(received().length).toEqual(0);
    });
});

describe('HeaptrackWrapper FD behavior tests', () => {
    it('should verify net.Socket properly handles external FDs', async () => {
        // This test verifies that net.Socket can handle external FDs and properly goes through its close sequence.
        // This is critical for HeaptrackWrapper which creates sockets from pipe FDs.

        if (process.platform === 'win32') {
            return;
        }

        const tmpDir = os.tmpdir();
        const pipePath = path.join(tmpDir, `test_pipe_${process.pid}_${Date.now()}`);

        try {
            // Create a named pipe (exactly like HeaptrackWrapper does)
            execSync(`mkfifo "${pipePath}"`);

            // Open the pipe with O_NONBLOCK (like HeaptrackWrapper does)
            const O_NONBLOCK = fs.constants.O_NONBLOCK; // Use fs.constants for portability
            const O_RDWR = fs.constants.O_RDWR;
            const fd = fs.openSync(pipePath, O_RDWR | O_NONBLOCK);

            // Create a net.Socket with the pipe FD (like HeaptrackWrapper)
            const socket = new net.Socket({fd: fd, readable: true, writable: true});

            // Set up close event listener
            const closePromise = new Promise<void>(resolve => {
                socket.on('close', () => {
                    // Close event fired - this means the socket went through its complete cleanup sequence
                    resolve();
                });
            });

            // Destroy the socket (like HeaptrackWrapper does)
            socket.destroy();

            // Wait for close event to fire
            await closePromise;

            // Success! The socket properly handled the external FD and completed its close sequence. We assume
            // (based on manual testing) that this includes closing the FD. We NEVER attempt to verify the FD
            // state ourselves to avoid FD recycling race conditions.
        } finally {
            // Clean up the pipe
            try {
                fs.unlinkSync(pipePath);
            } catch {
                // Ignore cleanup errors
            }
        }
    });
});

function execResult(overrides: Partial<UnprocessedExecResult> = {}): UnprocessedExecResult {
    return {
        code: 0,
        okToCache: false,
        filenameTransform: x => x,
        stdout: '',
        stderr: '',
        execTime: 0,
        timedOut: false,
        truncated: false,
        ...overrides,
    };
}

// Stands in for the sandboxed program: writes `bytes` into the heaptrack fifo, as the preload library would.
async function writeToFifo(fifo: string, bytes: number) {
    const fd = fs.openSync(fifo, fs.constants.O_WRONLY | fs.constants.O_NONBLOCK);
    const buffer = Buffer.alloc(64 * 1024, 'x');
    let gotEpipe = false;
    try {
        let left = bytes;
        while (left > 0) {
            const n = Math.min(left, buffer.length);
            try {
                fs.writeSync(fd, buffer, 0, n);
                left -= n;
            } catch (e: any) {
                if (e.code === 'EPIPE') {
                    gotEpipe = true;
                    break;
                }
                if (e.code !== 'EAGAIN') throw e;
            }
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    } finally {
        fs.closeSync(fd);
    }
    // Give the reader time to drain the fifo, as a real process exit would.
    await new Promise(resolve => setTimeout(resolve, 50));
    return gotEpipe;
}

function makeWrapper(
    maxRawOutput: number,
    bytesWritten: number,
    interpretResult = execResult({stdout: 'interpreted'}),
) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'heaptrack-test-'));
    const executed: string[] = [];
    const state = {gotEpipe: false};
    const sandbox = async (_file: string, _args: string[], options: ExecutionOptions) => {
        state.gotEpipe = await writeToFifo(options.env!.DUMP_HEAPTRACK_OUTPUT, bytesWritten);
        return execResult({stderr: 'program stderr'});
    };
    const exec = async (file: string) => {
        executed.push(path.basename(file));
        return file.endsWith('heaptrack_interpret') ? interpretResult : execResult();
    };
    const props = (key: string, defaultValue?: any) => {
        if (key === 'heaptrackMaxRawOutput') return maxRawOutput;
        if (key === 'heaptrackPath') return '/heaptrack';
        return defaultValue;
    };
    const wrapper = new HeaptrackWrapper(dir, sandbox, exec, [], props as any, 'none');
    return {wrapper, dir, executed, state};
}

describe.skipIf(process.platform === 'win32')('HeaptrackWrapper trace size limit', () => {
    it('interprets a trace within the limit', {timeout: 10000}, async () => {
        const {wrapper, dir, executed, state} = makeWrapper(1024 * 1024, 300 * 1024);
        const result = await wrapper.exec('program', [], {});
        expect(result.stderr).toEqual('program stderr');
        expect(state.gotEpipe).toBe(false);
        expect(fs.statSync(path.join(dir, 'heaptrack_raw.txt')).size).toBeGreaterThanOrEqual(300 * 1024);
        expect(executed).toEqual(['heaptrack_interpret', 'heaptrack_print']);
        fs.rmSync(dir, {recursive: true});
    });

    it('discards a trace over the limit and stops the writer', {timeout: 10000}, async () => {
        const {wrapper, dir, executed, state} = makeWrapper(1024 * 1024, 20 * 1024 * 1024);
        const result = await wrapper.exec('program', [], {});
        expect(result.stderr).toContain('Heaptrack trace exceeded 1 MiB and was discarded');
        expect(state.gotEpipe).toBe(true);
        expect(fs.existsSync(path.join(dir, 'heaptrack_raw.txt'))).toBe(false);
        expect(executed).toEqual([]);
        fs.rmSync(dir, {recursive: true});
    });

    it('reports a limit that is not a whole number of MiB in bytes', {timeout: 10000}, async () => {
        const {wrapper, dir} = makeWrapper(600 * 1024, 1024 * 1024);
        const result = await wrapper.exec('program', [], {});
        expect(result.stderr).toContain('exceeded 614400 bytes');
        fs.rmSync(dir, {recursive: true});
    });

    it('does not print a truncated interpreter output', {timeout: 10000}, async () => {
        const {wrapper, dir, executed} = makeWrapper(
            1024 * 1024,
            1024,
            execResult({stdout: 'partial', truncated: true}),
        );
        const result = await wrapper.exec('program', [], {});
        expect(result.stderr).toContain('Interpreted heaptrack trace exceeded 1 MiB');
        expect(executed).toEqual(['heaptrack_interpret']);
        fs.rmSync(dir, {recursive: true});
    });
});
