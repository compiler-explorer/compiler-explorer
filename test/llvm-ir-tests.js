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

const chai = require('chai');
const chaiAsPromised = require("chai-as-promised");
const LLCCompiler = require('../lib/compilers/llc');
const OPTCompiler = require('../lib/compilers/opt');
const CompilationEnvironment = require('../lib/compilation-env');
const properties = require('../lib/properties');

chai.use(chaiAsPromised);
chai.should();


const languages = {
    llvm: {id: 'llvm'}
};

const compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));

function createCompiler(compiler) {
    const ce = new CompilationEnvironment(compilerProps);
    const info = {
        exe: null,
        remote: true,
        lang: languages.llvm.id
    };

    return new compiler(info, ce);
}

describe('llc options for at&t assembly', function () {
    let compiler = createCompiler(LLCCompiler);

    compiler.optionsForFilter({
        'intel': false,
        'binary': false
    }, 'output.s').should.eql(['-o', 'output.s']);
});

describe('llc options for intel assembly', function () {
    let compiler = createCompiler(LLCCompiler);

    compiler.optionsForFilter({
        'intel': true,
        'binary': false
    }, 'output.s').should.eql(['-o', 'output.s', '-x86-asm-syntax=intel']);
});

describe('llc options for at&t binary', function () {
    let compiler = createCompiler(LLCCompiler);

    compiler.optionsForFilter({
        'intel': false,
        'binary': true
    }, 'output.s').should.eql(['-o', 'output.s', '-filetype=obj']);
});

describe('llc options for intel binary', function () {
    let compiler = createCompiler(LLCCompiler);

    compiler.optionsForFilter({
        'intel': true,
        'binary': true
    }, 'output.s').should.eql(['-o', 'output.s', '-filetype=obj']);
});

describe('opt options', function () {
    let compiler = createCompiler(OPTCompiler);

    compiler.optionsForFilter({
        'intel': false,
        'binary': false
    }, 'output.s').should.eql(['-o', 'output.s', '-S']);
});
