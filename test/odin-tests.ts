// Copyright (c) 2024, Compiler Explorer Authors
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

import {beforeAll, describe, expect, it} from 'vitest';

import {OdinCompiler} from '../lib/compilers/odin.js';
import {CompilerOutputOptions} from '../types/features/filters.interfaces.js';
import {LanguageKey} from '../types/languages.interfaces.js';

import {fs, makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    odin: {id: 'odin' as LanguageKey},
};

let ce;
const info = {
    exe: '/dev/null',
    remote: {
        target: 'example',
        path: 'dummy',
        cmakePath: 'cmake',
        basePath: '/',
    },
    lang: languages.odin.id,
};

describe('Odin source preprocessing tests', () => {
    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('Test @require attribute is added correctly', async () => {
        const compiler = new OdinCompiler(makeFakeCompilerInfo(info), ce);
        const inputFile = 'test/odin/add_require.odin';
        const filter: CompilerOutputOptions = {};
        const output = compiler.preProcess(fs.readFileSync(inputFile).toString(), filter);
        const expectedOutputFile = 'test/odin/add_require.odin.expected';
        const expectedOutput = fs.readFileSync(expectedOutputFile).toString();
        expect(output).toEqual(expectedOutput);
    });

    it('Test source unmodified with binary or execute filter', async () => {
        const compiler = new OdinCompiler(makeFakeCompilerInfo(info), ce);
        const inputFile = 'test/odin/add_require.odin';
        const filter: CompilerOutputOptions = {execute: true};
        const output = compiler.preProcess(fs.readFileSync(inputFile).toString(), filter);
        const expectedOutputFile = 'test/odin/add_require.odin';
        const expectedOutput = fs.readFileSync(expectedOutputFile).toString();
        expect(output).toEqual(expectedOutput);
    });
});
