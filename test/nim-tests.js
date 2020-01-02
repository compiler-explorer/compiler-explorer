// Copyright (c) 2019, Bastien Penavayre
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
const NimCompiler = require('../lib/compilers/nim');
const CompilationEnvironment = require('../lib/compilation-env');
const properties = require('../lib/properties');

chai.use(chaiAsPromised);
chai.should();

const languages = {
    nim: {id: 'nim'}
};

const compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));

describe('Nim', () => {
    const ce = new CompilationEnvironment(compilerProps);
    const info = {
        exe: null,
        remote: true,
        lang: languages.nim.id
    };

    it('Nim should not allow --run/-r parameter', () => {
        const compiler = new NimCompiler(info, ce);
        compiler.filterUserOptions(["c", "--run", "--something"]).should.deep.equal(["c", "--something"]);
        compiler.filterUserOptions(["cpp", "-r", "--something"]).should.deep.equal(["cpp", "--something"]);
    });

    it('Nim compile to Cpp if not asked otherwise', () => {
        const compiler = new NimCompiler(info, ce);
        compiler.filterUserOptions([]).should.deep.equal(["compile"]);
        compiler.filterUserOptions(["badoption"]).should.deep.equal(["compile", "badoption"]);
        compiler.filterUserOptions(["js"]).should.deep.equal(["js"]);
    });

    it("test getCacheFile from possible user-options", () => {
        const compiler = new NimCompiler(info, ce),
            input = "test.min",
            folder = "/tmp/",
            expected = {
                "cpp": folder + '@m' + input + ".cpp.o",
                "c": folder + '@m' + input + ".c.o",
                "objc": folder + '@m' + input + ".m.o",
            };

        for (const lang of ["cpp", "c", "objc"]) {
            compiler.getCacheFile([lang], input, folder).should.equal(expected[lang]);
        }
        chai.assert.equal(compiler.getCacheFile([], input, folder), null);
        chai.assert.equal(compiler.getCacheFile(["js"], input, folder), null);
    });
});
