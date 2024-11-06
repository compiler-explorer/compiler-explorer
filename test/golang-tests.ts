// Copyright (c) 2018, Compiler Explorer Authors
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

import {GolangCompiler} from '../lib/compilers/golang.js';
import * as utils from '../lib/utils.js';
import {LanguageKey} from '../types/languages.interfaces.js';

import {fs, makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    go: {id: 'go' as LanguageKey},
};

let ce;
const info = {
    exe: '/dev/null',
    remote: {
        target: 'foo',
        path: 'bar',
        cmakePath: 'cmake',
        basePath: '/',
    },
    lang: languages.go.id,
};

async function testGoAsm(baseFilename: string) {
    const compiler = new GolangCompiler(makeFakeCompilerInfo(info), ce);

    const asmLines = utils.splitLines(fs.readFileSync(baseFilename + '.asm').toString());

    const result = {
        stderr: asmLines.map(line => {
            return {
                text: line,
            };
        }),
    };

    const [output] = await compiler.postProcess(result);
    const expectedOutput = utils.splitLines(fs.readFileSync(baseFilename + '.output.asm').toString());
    expect(utils.splitLines(output.asm)).toEqual(expectedOutput);
    expect(output).toEqual({
        asm: expectedOutput.join('\n'),
        stdout: [],
        stderr: [],
    });
}

describe('GO asm tests', () => {
    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('Handles unknown line number correctly', async () => {
        await testGoAsm('test/golang/bug-901');
    });
    it('Rewrites PC jumps to labels', async () => {
        await testGoAsm('test/golang/labels');
    });
});
