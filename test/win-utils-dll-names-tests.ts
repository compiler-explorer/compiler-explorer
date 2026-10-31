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

import fs from 'node:fs/promises';
import path from 'node:path';

import {describe, expect, it} from 'vitest';

import {WinUtils} from '../lib/binaries/win-utils.js';
import {newTempDir} from './utils.js';

describe('WinUtils.get_dlls_used import names', () => {
    it('only looks up imports that are plain file names', async () => {
        const root = newTempDir();
        const pathDir = path.join(root, 'bin');
        const exe = path.join(root, 'output.exe');
        await fs.mkdir(pathDir, {recursive: true});
        await fs.writeFile(exe, '');
        await fs.writeFile(path.join(pathDir, 'plain.dll'), '');
        await fs.writeFile(path.join(root, 'outside.dll'), '');

        const names = [
            'plain.dll',
            '../outside.dll',
            '..\\outside.dll',
            'sub/outside.dll',
            'C:\\Windows\\outside.dll',
            'C:outside.dll',
            '\\\\host\\share\\outside.dll',
        ];
        const exec = async (_cmd: string, args: string[]) => ({
            stdout: args[1] === exe ? names.map(name => `\tDLL Name: ${name}`).join('\n') : '',
        });
        const winUtils = new WinUtils(exec as any, 'objdump', {env: {PATH: pathDir}} as any);

        expect(await winUtils.get_dlls_used(exe)).toEqual([path.join(pathDir, 'plain.dll')]);
    });
});
