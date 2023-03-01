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

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {AnalysisTool, LLVMmcaTool} from '../lib/compilers/index.js';

import {
    makeCompilationEnvironment,
    makeFakeCompilerInfo,
    makeFakeParseFiltersAndOutputOptions,
    shouldExist,
} from './utils.js';

const languages = {
    analysis: {id: 'analysis'},
} as const;

describe('LLVM-mca tool definition', () => {
    let ce: CompilationEnvironment;
    let a: LLVMmcaTool;

    before(() => {
        ce = makeCompilationEnvironment({languages});
        const info = makeFakeCompilerInfo({
            remote: {
                target: 'foo',
                path: 'bar',
            },
            lang: languages.analysis.id,
        });
        a = new LLVMmcaTool(info, ce);
    });

    it('should have most filters disabled', () => {
        if (shouldExist(a)) {
            a.getInfo().disabledFilters.should.be.deep.equal(['labels', 'directives', 'commentOnly', 'trim']);
        }
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
        const opts = a.optionsForFilter(
            makeFakeParseFiltersAndOutputOptions({
                commentOnly: false,
                labels: true,
            }),
            'output.txt',
        );
        opts.should.be.deep.equal(['-o', 'output.txt']);
    });

    it('should split if disabledFilters is a string', () => {
        const info = makeFakeCompilerInfo({
            remote: {
                target: 'foo',
                path: 'bar',
            },
            lang: 'analysis',
            disabledFilters: 'labels,directives' as any,
        });
        new AnalysisTool(info, ce).getInfo().disabledFilters.should.deep.equal(['labels', 'directives']);
    });
});
