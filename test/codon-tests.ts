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
import {CodonCompiler} from '../lib/compilers/codon.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    python: {id: 'python' as LanguageKey},
};

describe('Codon', () => {
    let compiler: CodonCompiler;

    beforeAll(() => {
        const ce: CompilationEnvironment = makeCompilationEnvironment({languages});
        compiler = new CodonCompiler(makeFakeCompilerInfo({exe: '/dev/null', lang: languages.python.id}), ce);
    });

    it('asks for an executable explicitly, as newer Codon infers assembly from the .s output name', () => {
        const filters = {};
        expect(compiler.optionsForFilter(filters, 'output.s', [])).toEqual(['build', '-exe', '-o', 'output.s']);
        expect(filters).toEqual({binary: true});
    });

    it('leaves the output kind to -llvm when the user asks for LLVM IR', () => {
        const filters = {};
        expect(compiler.optionsForFilter(filters, 'output.s', ['-llvm'])).toEqual(['build', '-o', 'output.s']);
        expect(filters).toEqual({binary: false});
    });
});
