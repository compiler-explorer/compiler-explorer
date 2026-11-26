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

import {beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {FlangCompiler} from '../lib/compilers/flang.js';
import {FlangFC1Compiler} from '../lib/compilers/flang-fc1.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    fortran: {id: 'fortran' as LanguageKey},
};

describe('Fortran', () => {
    let ce: CompilationEnvironment;
    const info = {
        exe: '/dev/null',
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
            basePath: '/',
        },
        lang: languages.fortran.id,
    };

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it.each([
        ['flang', FlangCompiler],
        ['flang-fc1', FlangFC1Compiler],
    ])('%s should not offer the Clang AST view, even when the version string mentions Clang', (_key, compilerClass) => {
        const compiler = new compilerClass(makeFakeCompilerInfo(info), ce);
        expect(compiler.couldSupportASTDump('flang version 20.1.8')).toEqual(false);
        expect(
            compiler.couldSupportASTDump(
                'AMD flang version 21.1.8 (CLANG: AOCC_6.0.0-Build#1942-gbc962a7cfe0a 2026_10_09)',
            ),
        ).toEqual(false);
        expect(
            compiler.couldSupportASTDump(
                'AMD aof version 21.1.8 (CLANG: AOCC_6.0.0-Build#1942-gbc962a7cfe0a 2026_10_09)',
            ),
        ).toEqual(false);
    });
});
