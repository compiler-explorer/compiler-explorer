// Copyright (c) 2018, Adrian Bibby Walther
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

import {ClangCompiler, ClangHexagonCompiler} from '../lib/compilers/clang.js';
import {DefaultCompiler} from '../lib/compilers/default.js';
import {LLCCompiler} from '../lib/compilers/llc.js';
import {OptCompiler} from '../lib/compilers/opt.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    'c++': {id: 'c++'},
    llvm: {id: 'llvm'},
};

let ce;

function createCompiler(compiler) {
    if (ce === undefined) {
        ce = makeCompilationEnvironment({languages});
    }

    const info = {
        exe: null,
        remote: true,
        lang: languages.llvm.id,
    };

    return new compiler(info, ce);
}

describe('LLVM IR Compiler', () => {
    let compiler;

    beforeAll(() => {
        compiler = createCompiler(LLCCompiler);
    });

    it('llc options for at&t assembly', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: false,
                    binary: false,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s']);
    });

    it('llc options for intel assembly', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: true,
                    binary: false,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-x86-asm-syntax=intel']);
    });

    it('llc options for at&t binary', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: false,
                    binary: true,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-filetype=obj']);
    });

    it('llc options for intel binary', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: true,
                    binary: true,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-filetype=obj']);
    });

    it('opt options', () => {
        const compiler = createCompiler(OptCompiler);

        expect(
            compiler.optionsForFilter(
                {
                    intel: false,
                    binary: false,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-S']);
    });
});

describe('LLVM IR compilers and the Clang AST view', () => {
    const clangVersion = 'clang version 21.1.0 (https://github.com/llvm/llvm-project.git)';

    function createClangLikeCompiler(compiler, lang: LanguageKey) {
        if (ce === undefined) {
            ce = makeCompilationEnvironment({languages});
        }

        const info = makeFakeCompilerInfo({
            exe: '/dev/null',
            remote: {
                target: 'foo',
                path: 'bar',
                cmakePath: 'cmake',
                basePath: '/',
            },
            lang: lang,
        });

        return new compiler(info, ce);
    }

    it('should not offer the AST view for clang compiling LLVM IR', () => {
        // irclang* compilers have no compilerType, irhexagonclang* use clang-hexagon
        expect(createClangLikeCompiler(DefaultCompiler, 'llvm').couldSupportASTDump(clangVersion)).toEqual(false);
        expect(createClangLikeCompiler(ClangHexagonCompiler, 'llvm').couldSupportASTDump(clangVersion)).toEqual(false);
    });

    it('should still offer the AST view for clang compiling C++', () => {
        expect(createClangLikeCompiler(ClangCompiler, 'c++').couldSupportASTDump(clangVersion)).toEqual(true);
        expect(createClangLikeCompiler(ClangHexagonCompiler, 'c++').couldSupportASTDump(clangVersion)).toEqual(true);
    });
});
