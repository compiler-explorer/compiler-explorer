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

const IMAGE_FILE_MACHINE_I386 = 0x14c;
const IMAGE_FILE_MACHINE_AMD64 = 0x8664;

async function writeFakePe(filename: string, machine: number) {
    const buf = Buffer.alloc(0x100);
    buf.write('MZ', 0, 'latin1');
    buf.writeUInt32LE(0x80, 0x3c);
    buf.write('PE\0\0', 0x80, 'latin1');
    buf.writeUInt16LE(machine, 0x84);
    await fs.mkdir(path.dirname(filename), {recursive: true});
    await fs.writeFile(filename, buf);
}

describe('WinUtils.get_dlls_used', () => {
    it('skips DLLs whose architecture does not match the executable', async () => {
        const root = newTempDir();
        const x64Dir = path.join(root, 'system32');
        const x86Dir = path.join(root, 'x86');
        const exe = path.join(root, 'output.exe');
        await writeFakePe(exe, IMAGE_FILE_MACHINE_I386);
        await writeFakePe(path.join(x64Dir, 'ucrtbased.dll'), IMAGE_FILE_MACHINE_AMD64);
        await writeFakePe(path.join(x86Dir, 'ucrtbased.dll'), IMAGE_FILE_MACHINE_I386);

        const exec = async (_cmd: string, args: string[]) => ({
            stdout: args[1] === exe ? '    DLL Name: ucrtbased.dll\n' : '',
        });
        const winUtils = new WinUtils(exec as any, 'objdump', {env: {PATH: [x64Dir, x86Dir].join(path.delimiter)}});

        expect(await winUtils.get_dlls_used(exe)).toEqual([path.join(x86Dir, 'ucrtbased.dll')]);
    });

    it('does not look up imports with a path in their name', async () => {
        const root = newTempDir();
        const pathDir = path.join(root, 'bin');
        const exe = path.join(root, 'output.exe');
        await writeFakePe(exe, IMAGE_FILE_MACHINE_I386);
        await writeFakePe(path.join(root, 'outside.dll'), IMAGE_FILE_MACHINE_I386);
        await fs.mkdir(pathDir, {recursive: true});

        const exec = async (_cmd: string, args: string[]) => ({
            stdout: args[1] === exe ? '    DLL Name: ../outside.dll\n' : '',
        });
        const winUtils = new WinUtils(exec as any, 'objdump', {env: {PATH: pathDir}});

        expect(await winUtils.get_dlls_used(exe)).toEqual([]);
    });
});

describe('WinUtils.get_dll_names', () => {
    it('only returns plain file names', () => {
        const objdump_output = [
            '\tDLL Name: KERNEL32.dll',
            '\tDLL Name: ucrtbased.dll',
            '\tDLL Name: ..\\..\\outside.dll',
            '\tDLL Name: ../outside.dll',
            '\tDLL Name: sub/outside.dll',
            '\tDLL Name: C:\\Windows\\outside.dll',
            '\tDLL Name: C:outside.dll',
            '\tDLL Name: \\\\host\\share\\outside.dll',
            '\tvma:  Hint/Ord Member-Name Bound-To',
        ].join('\n');
        const winUtils = new WinUtils((async () => ({stdout: ''})) as any, 'objdump', {env: {PATH: ''}} as any);

        expect(winUtils.get_dll_names(objdump_output)).toEqual(['KERNEL32.dll', 'ucrtbased.dll']);
    });
});
