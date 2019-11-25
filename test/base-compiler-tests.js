// Copyright (c) 2018, Compiler Explorer Authors
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
const sinon = require('sinon');
const chaiAsPromised = require("chai-as-promised");
const BaseCompiler = require('../lib/base-compiler');
const CompilationEnvironment = require('../lib/compilation-env');
const properties = require('../lib/properties');
const fs = require('fs');

chai.use(chaiAsPromised);
const should = chai.should();

const languages = {
    'c++': {id: 'c++'}
};

const compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));

describe('Basic compiler invariants', function () {
    const ce = new CompilationEnvironment(compilerProps);
    const info = {
        exe: null,
        remote: true,
        lang: languages['c++'].id
    };

    const compiler = new BaseCompiler(info, ce);

    it('should recognize when optOutput has been request', () => {
        compiler.optOutputRequested(["please", "recognize", "-fsave-optimization-record"]).should.equal(true);
        compiler.optOutputRequested(["please", "don't", "recognize"]).should.equal(false);
    });
    // Overkill test, but now we're safer!
    it('should recognize cfg compilers', () => {
        compiler.isCfgCompiler("clang version 5.0.0 (https://github.com/asutton/clang.git 449c8c3e91355a3b2b6761e11d9fb5d3c125b791) (https://github.com/llvm-mirror/llvm.git 40b1e969f9cb2a0c697e247435193fb006ef1311)").should.equal(true);
        compiler.isCfgCompiler("clang version 4.0.0 (tags/RELEASE_400/final 299826)").should.equal(true);
        compiler.isCfgCompiler("clang version 7.0.0 (trunk 325868)").should.equal(true);
        compiler.isCfgCompiler("clang version 3.3 (tags/RELEASE_33/final)").should.equal(true);
        compiler.isCfgCompiler("clang version 6.0.0 (tags/RELEASE_600/final 327031) (llvm/tags/RELEASE_600/final 327028)").should.equal(true);

        compiler.isCfgCompiler("g++ (GCC-Explorer-Build) 4.9.4").should.equal(true);
        compiler.isCfgCompiler("g++ (GCC-Explorer-Build) 8.0.1 20180223 (experimental)").should.equal(true);
        compiler.isCfgCompiler("g++ (GCC-Explorer-Build) 8.0.1 20180223 (experimental)").should.equal(true);
        compiler.isCfgCompiler("g++ (GCC) 4.1.2").should.equal(true);

        compiler.isCfgCompiler("foo-bar-g++ (GCC-Explorer-Build) 4.9.4").should.equal(true);
        compiler.isCfgCompiler("foo-bar-gcc (GCC-Explorer-Build) 4.9.4").should.equal(true);
        compiler.isCfgCompiler("foo-bar-gdc (GCC-Explorer-Build) 4.9.4").should.equal(true);

        compiler.isCfgCompiler("fake-for-test (Based on g++)").should.equal(false);

        compiler.isCfgCompiler("gdc (crosstool-NG 203be35 - 20160205-2.066.1-e95a735b97) 5.2.0").should.equal(true);
        compiler.isCfgCompiler("gdc (crosstool-NG hg+unknown-20131212.080758 - 20140430-2.064.2-404a037d26) 4.8.2").should.equal(true);
        compiler.isCfgCompiler("gdc (crosstool-NG crosstool-ng-1.20.0-232-gc746732 - 20150830-2.066.1-d0dd4a83de) 4.9.3").should.equal(true);

        compiler.isCfgCompiler("fake-for-test (Based on gdc)").should.equal(false);
    });
    it('should allow comments next to includes (Bug #874)', () => {
        should.equal(compiler.checkSource("#include <cmath> // std::(sin, cos, ...)"), null);
        const badSource = compiler.checkSource("#include </dev/null..> //Muehehehe");
        should.exist(badSource);
        badSource.should.equal("<stdin>:1:1: no absolute or relative includes please");
    });

    it('should compile', async () => {
        const execStub = sinon.stub(compiler, 'exec');
        execStub.callsFake((compiler, args, options) => {
            const minusO = args.indexOf("-o");
            minusO.should.be.gte(0);
            const output = args[minusO + 1];
            // Maybe we should mock out the FS too; but that requires a lot more work.
            fs.writeFileSync(output, "This is the output file");
            return Promise.resolve({
                code: 0,
                okToCache: true,
                filenameTransform: x => x,
                stdout: 'stdout',
                stderr: 'stderr'
            });
        });
        const result = await compiler.compile(
            "source",
            "options",
            {},
            {},
            false,
            [],
            {},
            []);
        result.code.should.equal(0);
        result.compilationOptions.should.contain("options");
        result.compilationOptions.should.contain(result.inputFilename);
        result.okToCache.should.be.true;
        result.asm.should.deep.equal([{source: null, text: "This is the output file"}]);
        result.stdout.should.deep.equal([{text: "stdout"}]);
        result.stderr.should.deep.equal([{text: "stderr"}]);
        result.popularArguments.should.deep.equal({});
        result.tools.should.deep.equal([]);
        execStub.called.should.be.true;
        execStub.restore();
    });
});
