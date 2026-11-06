// Copyright (c) 2019, Compiler Explorer Authors
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
import {randomBytes} from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';

import {describe, expect, it, vi} from 'vitest';

import {Packager} from '../lib/packager.js';
import {newTempDir} from './utils.js';

async function writeTestFile(filepath: string) {
    return fs.writeFile(filepath, '#!/bin/sh\n\necho Hello, world!\n\n');
}

describe('Packager', () => {
    it('should be able to package 1 file', async () => {
        const pack = new Packager();

        const dirPath = newTempDir();
        await writeTestFile(path.join(dirPath, 'hello.txt'));

        const targzPath = path.join(newTempDir(), 'package.tgz');
        await pack.package(dirPath, targzPath);

        await expect(fs.stat(targzPath)).resolves.toBeDefined();
    }, 5000);

    it('should be able to unpack', async () => {
        const pack = new Packager();

        const dirPath = newTempDir();
        await writeTestFile(path.join(dirPath, 'hello.txt'));

        const targzPath = path.join(newTempDir(), 'package.tgz');
        await pack.package(dirPath, targzPath);

        const unpackPath = newTempDir();
        const pack2 = new Packager();
        await pack2.unpack(targzPath, unpackPath);

        const unpackedFilepath = path.join(unpackPath, 'hello.txt');
        await expect(fs.stat(unpackedFilepath)).resolves.toBeDefined();
    }, 5000);

    it.skipIf(process.platform === 'win32')(
        'should keep file modes and follow symlinks',
        async () => {
            const pack = new Packager();

            const dirPath = newTempDir();
            await writeTestFile(path.join(dirPath, 'hello.sh'));
            await fs.chmod(path.join(dirPath, 'hello.sh'), 0o755);
            await fs.symlink('hello.sh', path.join(dirPath, 'link.sh'));

            const targzPath = path.join(newTempDir(), 'package.tgz');
            await pack.package(dirPath, targzPath);
            const unpackPath = newTempDir();
            await pack.unpack(targzPath, unpackPath);

            const unpackedMode = (await fs.stat(path.join(unpackPath, 'hello.sh'))).mode & 0o777;
            expect(unpackedMode).toEqual(0o755);
            expect((await fs.lstat(path.join(unpackPath, 'link.sh'))).isSymbolicLink()).toBe(false);
        },
        5000,
    );

    // A FIFO that has a holder but no data blocks the stream for real, in a filesystem call, with no error.
    describe.skipIf(process.platform === 'win32')('when a stream stalls', () => {
        async function withFifo(fifoPath: string, fn: () => Promise<void>) {
            execFileSync('mkfifo', [fifoPath]);
            const holder = await fs.open(fifoPath, 'r+');
            try {
                await fn();
            } finally {
                // Closing the last other end is what lets the blocked read or write return.
                await holder.close();
            }
        }

        it('should reject when unpacking never gets any data', async () => {
            const fifoPath = path.join(newTempDir(), 'package.tgz');
            await withFifo(fifoPath, async () => {
                await expect(new Packager(100).unpack(fifoPath, newTempDir())).rejects.toThrow(/timed out after 100ms/);
            });
        }, 5000);

        it('should reject and remove the output when packing can never finish writing', async () => {
            const dirPath = newTempDir();
            // Incompressible and larger than a pipe's buffer, so the write blocks.
            await fs.writeFile(path.join(dirPath, 'random.bin'), randomBytes(1024 * 1024));

            const fifoPath = path.join(newTempDir(), 'package.tgz');
            await withFifo(fifoPath, async () => {
                await expect(new Packager(100).package(dirPath, fifoPath)).rejects.toThrow(/timed out after 100ms/);
                await vi.waitFor(() => expect(fs.stat(fifoPath)).rejects.toThrow(/ENOENT/));
            });
        }, 5000);
    });
});
