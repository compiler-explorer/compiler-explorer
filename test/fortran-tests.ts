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
import {FlangCompiler, parseTreeArguments, parseTreeOutput} from '../lib/compilers/flang.js';
import {FlangFC1Compiler} from '../lib/compilers/flang-fc1.js';
import {splitArguments} from '../shared/common-utils.js';
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
    ])('%s offers the parse tree as its AST view for LLVM Flang only', (_key, compilerClass) => {
        const compiler = new compilerClass(makeFakeCompilerInfo(info), ce);
        for (const version of [
            'flang-new version 16.0.6 (https://github.com/llvm/llvm-project.git 7cbf1a2591520c2491aa35339f227775f4d3adf6)',
            'flang version 20.1.8 (https://github.com/llvm/llvm-project.git 87f0227cb60147a26a1eeb4fb06e3b505e9c7261)',
            'flang version 24.0.0git (https://github.com/llvm/llvm-project.git 63f1a78d2dcad39ded4732ec14de34a66a1782d2)',
            'AMD aof version 21.1.8 (CLANG: AOCC_6.0.0-Build#1942-gbc962a7cfe0a 2026_10_09)',
        ]) {
            expect(compiler.couldSupportASTDump(version)).toEqual(true);
        }
        // AOCC's classic Flang
        for (const version of [
            'AMD clang version 16.0.3 (CLANG: AOCC_4.1.0-Build#270 2023_07_10)',
            'AMD flang version 21.1.8 (CLANG: AOCC_6.0.0-Build#1942-gbc962a7cfe0a 2026_10_09)',
        ]) {
            expect(compiler.couldSupportASTDump(version)).toEqual(false);
        }
    });

    it('turns the frontend command from `flang -###` into a parse tree dump', () => {
        const fc1Line =
            ' "/opt/flang/bin/flang-24" "-fc1" "-triple" "x86_64-unknown-linux-gnu" "-S" "-cpp" "-fopenmp"' +
            ' "-fintrinsic-modules-path" "/opt/flang/finclude" "-O2" "-o" "/tmp/out dir/output.s" "-x" "f95" "/tmp/example.f90"';
        expect(parseTreeArguments(splitArguments(fc1Line).slice(1))).toEqual([
            '-fc1',
            '-triple',
            'x86_64-unknown-linux-gnu',
            '-cpp',
            '-fopenmp',
            '-fintrinsic-modules-path',
            '/opt/flang/finclude',
            '-O2',
            '-x',
            'f95',
            '/tmp/example.f90',
            '-fdebug-dump-parse-tree',
        ]);
    });

    it('drops actions the user added to a `flang -fc1` command', () => {
        expect(parseTreeArguments(['-fc1', '-emit-llvm', '-fdebug-unparse', '-o', 'output.s', 'example.f90'])).toEqual([
            '-fc1',
            'example.f90',
            '-fdebug-dump-parse-tree',
        ]);
    });

    it('shows the parse tree without its banner, or the errors when there is none', () => {
        const result = {
            code: 0,
            stdout: [
                {text: '======================== Flang: parse tree dump ========================'},
                {text: 'Program -> ProgramUnit -> MainProgram'},
                {text: "| ProgramStmt -> Name = 'p'"},
            ],
            stderr: [],
        };
        expect(parseTreeOutput(result as any)).toEqual([
            {text: 'Program -> ProgramUnit -> MainProgram'},
            {text: "| ProgramStmt -> Name = 'p'"},
        ]);
        const failed = {code: 1, stdout: [], stderr: [{text: "error: No explicit type declared for 'x'"}]};
        expect(parseTreeOutput(failed as any)).toEqual([
            {text: '<Flang could not produce a parse tree>'},
            {text: "error: No explicit type declared for 'x'"},
        ]);
    });
});
