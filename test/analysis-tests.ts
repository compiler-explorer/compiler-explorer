// Copyright (c) 2018, Filipe Cabecinhas
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

import {LLVMmcaTool} from '../lib/compilers/llvm-mca';
import {CompilerInfo} from '../types/compiler.interfaces';
import {LanguageKey} from '../types/languages.interfaces';

import {makeCompilationEnvironment} from './utils';

const languages: {analysis: {id: LanguageKey}} = {
    analysis: {id: 'analysis'},
};

describe('LLVM-mca tool definition', () => {
    let ce, a;

    before(() => {
        ce = makeCompilationEnvironment({languages});
        const info: Partial<CompilerInfo> = {
            remote: true,
            lang: languages.analysis.id,
        };
        a = new LLVMmcaTool(info as CompilerInfo, ce);
    });

    it('should have most filters disabled', () => {
        a.compiler.disabledFilters.should.be.deep.equal(['labels', 'directives', 'commentOnly', 'trim']);
    });

    it('should default to most filters off', () => {
        const filters = a.getDefaultFilters();
        filters.intel.should.equal(true);
        filters.commentOnly.should.equal(false);
        filters.directives.should.equal(false);
        filters.labels.should.equal(false);
        filters.optOutput.should.equal(false);
    });

    it('should not support objdump', () => {
        a.supportsObjdump().should.equal(false);
    });

    it('should support "-o output-file" by default', () => {
        const opts = a.optionsForFilter({commentOnly: false, labels: true}, 'output.txt');
        opts.should.be.deep.equal(['-o', 'output.txt']);
    });

    // CompilerInfo.disabledFilters is now typed as string[] which doesn't allow string
    // Since CompilerInfo only a type (and not a class) I assume something in the
    // BaseComiler/AnalsysTool/LLVMmcaTool was doing the splitting logic
    //
    // I see two  options here:
    //   - Modify disableFilters type to be string|string[] and fix typing issue sother places
    //   - Remove this test because it seems like the intent is to only support string[] (which is less complicated)
    // it('should split if disabledFilters is a string', () => {
    //     const info: Partial<CompilerInfo> = {
    //         remote: true,
    //         lang: 'analysis',
    //         disabledFilters: 'labels,directives',
    //     };
    //     const a = new LLVMmcaTool(info as CompilerInfo, ce);
    //     a.compiler.disabledFilters.should.be.deep.equal(['labels', 'directives']);
    // });
});
