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
import os from 'node:os';
import path from 'node:path';

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {copyNeededDlls} from '../../lib/binaries/win-utils.js';
import {Win32MingWClang} from '../../lib/compilers/win32-mingw-clang.js';
import {Win32MingWGcc} from '../../lib/compilers/win32-mingw-gcc.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from '../utils.js';

vi.mock('../../lib/binaries/win-utils.js', () => ({copyNeededDlls: vi.fn()}));

describe.each([
    {name: 'MinGW gcc', Compiler: Win32MingWGcc},
    {name: 'MinGW clang', Compiler: Win32MingWClang},
])('$name running a project build', ({Compiler}) => {
    let dirPath: string;

    beforeEach(async () => {
        dirPath = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-mingw-'));
        await fs.mkdir(path.join(dirPath, 'build'));
    });

    afterEach(async () => {
        vi.clearAllMocks();
        await fs.rm(dirPath, {recursive: true, force: true});
    });

    function makeCompiler() {
        return new Compiler(
            makeFakeCompilerInfo({exe: 'g++.exe', lang: 'c++', objdumper: 'objdump.exe', ldPath: [], libPath: []}),
            makeCompilationEnvironment({languages: {'c++': {id: 'c++'}}}),
        );
    }

    it('puts the DLLs it needs next to the executable, wherever the build put it', async () => {
        const artifact = path.join(dirPath, 'build', 'demo.exe');
        await fs.writeFile(artifact, 'MZ');

        await makeCompiler().prepareArtifactExecution(artifact);

        expect(copyNeededDlls).toHaveBeenCalledWith(
            path.join(dirPath, 'build'),
            artifact,
            expect.anything(),
            'objdump.exe',
            // On Windows the sandbox only lets objdump read what is under its home, which customCwd gives it
            expect.objectContaining({customCwd: path.join(dirPath, 'build')}),
        );
    });

    it('leaves alone a build that produced nothing to run', async () => {
        await makeCompiler().prepareArtifactExecution(path.join(dirPath, 'build', 'demo.exe'));

        expect(copyNeededDlls).not.toHaveBeenCalled();
    });
});
