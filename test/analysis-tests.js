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

const LLVMmcaTool = require('../lib/compilers/llvm-mca');
const CompilationEnvironment = require('../lib/compilation-env');
const properties = require('../lib/properties');
require('chai').should();

const languages = {
    analysis: {id: 'analysis'}
};

const compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));

describe('LLVM-mca tool definition', () => {
    const ce = new CompilationEnvironment(compilerProps);
    const info = {
        exe: null,
        remote: true,
        lang: languages.analysis.id
    };
    const a = new LLVMmcaTool(info, ce);

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

    it('should split if disabledFilters is a string', () => {
        const info = {
            exe: null,
            remote: true,
            lang: "analysis",
            disabledFilters: 'labels,directives'
        };
        const a = new LLVMmcaTool(info, ce);
        a.compiler.disabledFilters.should.be.deep.equal(['labels', 'directives']);
    });
});
