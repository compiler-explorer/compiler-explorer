// Copyright (c) 2018, Rubén Rincón
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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ,
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

const chai = require('chai');
const chaiAsPromised = require("chai-as-promised");
const LDCCompiler = require('../lib/compilers/ldc');
const DMDCompiler = require('../lib/compilers/dmd');
const CompilationEnvironment = require('../lib/compilation-env');
const properties = require('../lib/properties');

chai.use(chaiAsPromised);
chai.should();

const languages = {
    d: {id: 'd'}
};

describe('D', () => {
    let ce;
    const info = {
        exe: null,
        remote: true,
        lang: languages.d.id
    };

    before(() => {
        const compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        ce = new CompilationEnvironment(compilerProps);
    });

    it('LDC should not allow -run parameter', () => {
        const compiler = new LDCCompiler(info, ce);
        compiler.filterUserOptions(["hello", "-run", "--something"]).should.deep.equal(["hello", "--something"]);
    });
    it('DMD should not allow -run parameter', () => {
        const compiler = new DMDCompiler(info, ce);
        compiler.filterUserOptions(["hello", "-run", "--something"]).should.deep.equal(["hello", "--something"]);
    });

    it('LDC supports AST output since version 1.4.0', () => {
        const compiler = new LDCCompiler(info, ce);
        compiler.couldSupportASTDump("LDC - the LLVM D compiler (1.3.0)").should.equal(false);
        compiler.couldSupportASTDump("LDC - the LLVM D compiler (1.4.0)").should.equal(true);
        compiler.couldSupportASTDump("LDC - the LLVM D compiler (1.8.0git-d54d25b-dirty)").should.equal(true);
        compiler.couldSupportASTDump("LDC - the LLVM D compiler (1.10.0)").should.equal(true);
    });
});
