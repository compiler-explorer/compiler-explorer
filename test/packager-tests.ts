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

import {describe, expect, it} from 'vitest';

import {Packager} from '../lib/packager.js';

import {fs, newTempDir, path} from './utils.js';

function writeTestFile(filepath) {
    return fs.writeFile(filepath, '#!/bin/sh\n\necho Hello, world!\n\n');
}

describe('Packager', () => {
    it(
        'should be able to package 1 file',
        async () => {
            const pack = new Packager();

            const dirPath = newTempDir();
            await writeTestFile(path.join(dirPath, 'hello.txt'));

            const targzPath = path.join(dirPath, 'package.tgz');
            await pack.package(dirPath, targzPath);

            await expect(fs.exists(targzPath)).resolves.toBe(true);
        },
        {timeout: 5000},
    );

    it(
        'should be able to unpack',
        async () => {
            const pack = new Packager();

            const dirPath = newTempDir();
            await writeTestFile(path.join(dirPath, 'hello.txt'));

            const targzPath = path.join(dirPath, 'package.tgz');
            await pack.package(dirPath, targzPath);

            const unpackPath = newTempDir();
            const pack2 = new Packager();
            await pack2.unpack(targzPath, unpackPath);

            const unpackedFilepath = path.join(unpackPath, 'hello.txt');
            await expect(fs.exists(unpackedFilepath)).resolves.toBe(true);
        },
        {timeout: 5000},
    );
});
