// Copyright (c) 2018, Patrick Quist
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
const PPCICompiler = require('../lib/compilers/ppci');
const {makeCompilationEnvironment} = require('./utils.js');

chai.use(chaiAsPromised);
chai.should();

const languages = {
    c: {id: 'c'}
};

describe('PPCI', function () {
    let ce;
    const info = {
        exe: null,
        remote: true,
        lang: languages.c.id
    };

    before(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('Should be ok with most arguments', () => {
        const compiler = new PPCICompiler(info, ce);
        compiler.filterUserOptions(["hello", "-help", "--something"]).should.deep.equal(["hello", "-help", "--something"]);
    });

    it('Should be ok with path argument', () => {
        const compiler = new PPCICompiler(info, ce);
        compiler.filterUserOptions(["hello", "--stuff", "/proc/cpuinfo"]).should.deep.equal(["hello", "--stuff", "/proc/cpuinfo"]);
    });

    it('Should be Not ok with report arguments', () => {
        const compiler = new PPCICompiler(info, ce);
        compiler.filterUserOptions(["hello", "--report", "--text-report", "--html-report"]).should.deep.equal(["hello"]);
    });
});
