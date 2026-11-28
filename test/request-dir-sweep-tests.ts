// Copyright (c) 2026, Compiler Explorer Authors
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

import {execFileSync} from 'node:child_process';
import fs from 'node:fs/promises';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

import {afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi} from 'vitest';

import * as exec from '../lib/exec.js';
import * as props from '../lib/properties.js';
import {
    exemptFromSweep,
    numActiveGates,
    RequestDirSweepError,
    requestDirFor,
    runThenSweep,
    sweepNonRegularEntries,
} from '../lib/request-dir-sweep.js';
import * as utils from '../lib/utils.js';

const describeOnPosix = process.platform === 'win32' ? describe.skip : describe;

function mkfifo(p: string) {
    execFileSync('mkfifo', [p]);
}

async function exists(p: string) {
    try {
        await fs.lstat(p);
        return true;
    } catch {
        return false;
    }
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>(r => {
        resolve = r;
    });
    return {promise, resolve};
}

async function makeRequestDir() {
    return await fs.mkdtemp(path.join(os.tmpdir(), 'ce-sweep-test-'));
}

describe('requestDirFor', () => {
    it('finds the request directory a bound directory lies in', () => {
        // Resolved so that Windows gets its drive letter and separators.
        const expected = path.resolve('/tmp/ce-abc');
        expect(requestDirFor('/tmp/ce-abc', '/tmp')).toEqual(expected);
        expect(requestDirFor('/tmp/ce-abc/build/sub', '/tmp')).toEqual(expected);
    });
    it('declines anything that is not inside a request directory', () => {
        expect(requestDirFor('/tmp', '/tmp')).toBeUndefined();
        expect(requestDirFor('/opt/compiler-explorer/gcc', '/tmp')).toBeUndefined();
        expect(requestDirFor('/tmp/../etc', '/tmp')).toBeUndefined();
    });
});

describeOnPosix('sweepNonRegularEntries', () => {
    let dir: string;
    beforeEach(async () => {
        dir = await makeRequestDir();
    });
    afterEach(async () => {
        await fs.rm(dir, {recursive: true, force: true});
    });

    it('removes FIFOs and sockets, keeping files, directories and symlinks', async () => {
        await fs.writeFile(path.join(dir, 'output.s'), 'asm');
        await fs.mkdir(path.join(dir, 'sub', 'deeper'), {recursive: true});
        mkfifo(path.join(dir, 'fifo'));
        mkfifo(path.join(dir, 'sub', 'deeper', 'fifo'));
        await fs.symlink('output.s', path.join(dir, 'link'));
        await fs.symlink('fifo', path.join(dir, 'link-to-fifo'));
        const server = net.createServer();
        await new Promise<void>(resolve => server.listen(path.join(dir, 'sub', 'socket'), resolve));
        try {
            await sweepNonRegularEntries(dir);
        } finally {
            server.close();
        }

        expect(await exists(path.join(dir, 'fifo'))).toBe(false);
        expect(await exists(path.join(dir, 'sub', 'deeper', 'fifo'))).toBe(false);
        expect(await exists(path.join(dir, 'sub', 'socket'))).toBe(false);
        expect(await fs.readFile(path.join(dir, 'output.s'), 'utf8')).toEqual('asm');
        expect(await fs.readlink(path.join(dir, 'link'))).toEqual('output.s');
        // Left dangling, which is harmless: it is the FIFO it pointed at that was the problem.
        expect(await fs.readlink(path.join(dir, 'link-to-fifo'))).toEqual('fifo');
    });

    it('does not descend through a symlink to a directory', async () => {
        const outside = await makeRequestDir();
        try {
            mkfifo(path.join(outside, 'fifo'));
            await fs.symlink(outside, path.join(dir, 'elsewhere'));
            await sweepNonRegularEntries(dir);
            expect(await exists(path.join(outside, 'fifo'))).toBe(true);
        } finally {
            await fs.rm(outside, {recursive: true, force: true});
        }
    });

    it('takes back a directory with no permissions and sweeps it', async () => {
        const locked = path.join(dir, 'locked');
        await fs.mkdir(locked);
        mkfifo(path.join(locked, 'fifo'));
        await fs.chmod(locked, 0o000);
        await sweepNonRegularEntries(dir);
        expect(await exists(path.join(locked, 'fifo'))).toBe(false);
    });

    it('takes back a directory whose parent can be listed but not searched', async () => {
        const parent = path.join(dir, 'a');
        await fs.mkdir(path.join(parent, 'b'), {recursive: true});
        mkfifo(path.join(parent, 'b', 'fifo'));
        await fs.chmod(parent, 0o600);
        await sweepNonRegularEntries(dir);
        expect(await exists(path.join(parent, 'b', 'fifo'))).toBe(false);
    });

    it('fails a tree with more entries than the limit', async () => {
        for (const name of ['a', 'b', 'c']) await fs.writeFile(path.join(dir, name), '');
        await expect(sweepNonRegularEntries(dir, {maxEntries: 2, maxDepth: 10})).rejects.toThrow(RequestDirSweepError);
        await expect(sweepNonRegularEntries(dir, {maxEntries: 3, maxDepth: 10})).resolves.toBeUndefined();
    });

    it('fails a tree nested deeper than the limit', async () => {
        await fs.mkdir(path.join(dir, 'a', 'b', 'c'), {recursive: true});
        await expect(sweepNonRegularEntries(dir, {maxEntries: 100, maxDepth: 2})).rejects.toThrow(RequestDirSweepError);
        await expect(sweepNonRegularEntries(dir, {maxEntries: 100, maxDepth: 3})).resolves.toBeUndefined();
    });

    it('is fine with a directory that has already gone', async () => {
        await expect(sweepNonRegularEntries(path.join(dir, 'missing'))).resolves.toBeUndefined();
    });

    it('keeps an exempted path until it is released', async () => {
        const fifo = path.join(dir, 'heaptrack_fifo');
        mkfifo(fifo);
        const release = exemptFromSweep(fifo);
        await sweepNonRegularEntries(dir);
        expect(await exists(fifo)).toBe(true);
        release();
        await sweepNonRegularEntries(dir);
        expect(await exists(fifo)).toBe(false);
    });
});

describeOnPosix('runThenSweep', () => {
    let dir: string;
    beforeEach(async () => {
        dir = await makeRequestDir();
    });
    afterEach(async () => {
        await fs.rm(dir, {recursive: true, force: true});
        expect(numActiveGates()).toEqual(0);
    });

    it('sweeps after a run and returns its result', async () => {
        const result = await runThenSweep(dir, async () => {
            mkfifo(path.join(dir, 'output.s'));
            return 'done';
        });
        expect(result).toEqual('done');
        expect(await exists(path.join(dir, 'output.s'))).toBe(false);
    });

    it('holds a result until every run in the directory has exited and the directory is swept', async () => {
        const second = deferred<void>();
        const events: string[] = [];

        const firstRun = runThenSweep(dir, async () => 'first').then(r => {
            events.push(r);
            return r;
        });
        const secondRun = runThenSweep(path.join(dir, 'build'), async () => {
            await second.promise;
            // Made by the run still going after the first has exited.
            mkfifo(path.join(dir, 'output.s'));
            return 'second';
        });

        // Let the first run exit; it must not be released while the second is still running.
        await new Promise(resolve => setTimeout(resolve, 50));
        expect(events).toEqual([]);

        second.resolve();
        expect(await firstRun).toEqual('first');
        expect(await secondRun).toEqual('second');
        // By the time either caller sees its result, the FIFO the second run made is gone.
        expect(await exists(path.join(dir, 'output.s'))).toBe(false);
    });

    it('still waits for the sweep when a run fails, and passes its error on', async () => {
        await expect(
            runThenSweep(dir, async () => {
                mkfifo(path.join(dir, 'output.s'));
                throw new Error('jail failed');
            }),
        ).rejects.toThrow('jail failed');
        expect(await exists(path.join(dir, 'output.s'))).toBe(false);
    });

    it('holds a run that arrives during a sweep until the sweep is done, and keeps earlier results held', async () => {
        const walkMayProceed = deferred<void>();
        const walkStarted = deferred<void>();
        const realOpendir = fs.opendir;
        const spy = vi.spyOn(fs, 'opendir').mockImplementation(async (...args: Parameters<typeof fs.opendir>) => {
            walkStarted.resolve();
            await walkMayProceed.promise;
            return realOpendir(...args);
        });
        try {
            const events: string[] = [];
            const first = runThenSweep(dir, async () => 'first').then(r => {
                events.push('first released');
                return r;
            });
            await walkStarted.promise;

            const fifo = path.join(dir, 'jobserver');
            const second = runThenSweep(dir, async () => {
                events.push('second started');
                // A FIFO the run makes and still needs must not be swept from under it.
                mkfifo(fifo);
                await new Promise(resolve => setTimeout(resolve, 20));
                events.push(`second sees its FIFO: ${await exists(fifo)}`);
                return 'second';
            });

            await new Promise(resolve => setTimeout(resolve, 20));
            expect(events).toEqual([]);

            walkMayProceed.resolve();
            expect(await second).toEqual('second');
            expect(await first).toEqual('first');
            expect(events).toEqual(['second started', 'second sees its FIFO: true', 'first released']);
            expect(await exists(fifo)).toBe(false);
        } finally {
            spy.mockRestore();
        }
    });

    it('reports a failed sweep in preference to the run failing', async () => {
        for (const name of ['a', 'b', 'c']) await fs.writeFile(path.join(dir, name), '');
        await expect(
            runThenSweep(
                dir,
                async () => {
                    throw new Error('jail failed');
                },
                {maxEntries: 1, maxDepth: 10},
            ),
        ).rejects.toThrow(RequestDirSweepError);
    });

    it('fails every waiting run if the directory cannot be swept', async () => {
        for (const name of ['a', 'b', 'c']) await fs.writeFile(path.join(dir, name), '');
        const limits = {maxEntries: 1, maxDepth: 10};
        const results = await Promise.allSettled([
            runThenSweep(dir, async () => 'a', limits),
            runThenSweep(dir, async () => 'b', limits),
        ]);
        for (const result of results) {
            expect(result.status).toEqual('rejected');
            expect((result as PromiseRejectedResult).reason).toBeInstanceOf(RequestDirSweepError);
        }
    });

    it('leaves a directory outside the temp root alone', async () => {
        // The current directory is the source tree; a sweep of it would be both slow and wrong.
        await expect(runThenSweep(process.cwd(), async () => 'ok')).resolves.toEqual('ok');
        await expect(runThenSweep(undefined, async () => 'ok')).resolves.toEqual('ok');
    });
});

// Runs the real exec.execute/exec.sandbox paths with a stand-in for nsjail that runs the command directly in the
// directory it was asked to bind, so a "jailed" command can leave a FIFO behind as hostile code in a real jail could.
describeOnPosix('nsjail runs sweep their directory', () => {
    let scriptDir: string;
    let dir: string;

    beforeAll(async () => {
        scriptDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-fake-nsjail-'));
        const script = path.join(scriptDir, 'nsjail');
        await fs.writeFile(
            script,
            [
                '#!/bin/sh',
                'while [ $# -gt 0 ]; do',
                '  case "$1" in',
                '    --bindmount) host=$(echo "$2" | cut -d: -f1); shift ;;',
                '    --) shift; break ;;',
                '  esac',
                '  shift',
                'done',
                'shift', // dumb-init
                'cd "$host" || exit 1',
                'exec "$@"',
                '',
            ].join('\n'),
        );
        await fs.chmod(script, 0o755);
        await fs.writeFile(
            path.join(scriptDir, 'execution.test.properties'),
            [
                `nsjail=${script}`,
                'executionType=nsjail',
                'sandboxType=nsjail',
                'nsjail.config.execute=execute.cfg',
                'nsjail.config.sandbox=sandbox.cfg',
                '',
            ].join('\n'),
        );
        props.initialize(scriptDir, ['test']);
    });
    afterAll(async () => {
        props.reset();
        await fs.rm(scriptDir, {recursive: true, force: true});
    });
    beforeEach(async () => {
        dir = await makeRequestDir();
    });
    afterEach(async () => {
        await fs.rm(dir, {recursive: true, force: true});
    });

    it('removes a FIFO left at the output path, so reading the output does not block', async () => {
        const result = await exec.execute('/bin/sh', ['-c', 'mkfifo output.s'], {customCwd: dir, timeoutMs: 5000});
        expect(result.code).toEqual(0);
        const output = path.join(dir, 'output.s');
        expect(await utils.fileExists(output)).toBe(false);
        await expect(fs.readFile(output)).rejects.toThrow(/ENOENT/);
    });

    it('removes a FIFO made by a run still going when another in the same directory exits', async () => {
        const [compile, view] = await Promise.all([
            exec.execute('/bin/sh', ['-c', 'echo asm > output.s'], {customCwd: dir, timeoutMs: 5000}),
            exec.sandbox('/bin/sh', ['-c', 'sleep 0.3; mkfifo output.ll'], {customCwd: dir, timeoutMs: 5000}),
        ]);
        expect(compile.code).toEqual(0);
        expect(view.code).toEqual(0);
        expect(await fs.readFile(path.join(dir, 'output.s'), 'utf8')).toEqual('asm\n');
        expect(await exists(path.join(dir, 'output.ll'))).toBe(false);
    });
});
