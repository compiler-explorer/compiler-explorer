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
const chaiAsPromised = require('chai-as-promised');
const BaseCompiler = require('../lib/base-compiler');
const fs = require('fs-extra');
const exec = require('../lib/exec');
const path = require('path');
const {makeCompilationEnvironment} = require('./utils');

chai.use(chaiAsPromised);
const should = chai.should();

const languages = {
    'c++': {id: 'c++'},
};

describe('Basic compiler invariants', function () {
    let ce, compiler;
    const info = {
        exe: null,
        remote: true,
        lang: languages['c++'].id,
        ldPath: [],
    };

    before(() => {
        ce = makeCompilationEnvironment({ languages });
        compiler = new BaseCompiler(info, ce);
    });

    it('should recognize when optOutput has been request', () => {
        compiler.optOutputRequested(['please', 'recognize', '-fsave-optimization-record']).should.equal(true);
        compiler.optOutputRequested(['please', "don't", 'recognize']).should.equal(false);
    });
    // Overkill test, but now we're safer!
    it('should recognize cfg compilers', () => {
        compiler.isCfgCompiler('clang version 5.0.0 (https://github.com/asutton/clang.git 449c8c3e91355a3b2b6761e11d9fb5d3c125b791) (https://github.com/llvm-mirror/llvm.git 40b1e969f9cb2a0c697e247435193fb006ef1311)').should.equal(true);
        compiler.isCfgCompiler('clang version 4.0.0 (tags/RELEASE_400/final 299826)').should.equal(true);
        compiler.isCfgCompiler('clang version 7.0.0 (trunk 325868)').should.equal(true);
        compiler.isCfgCompiler('clang version 3.3 (tags/RELEASE_33/final)').should.equal(true);
        compiler.isCfgCompiler('clang version 6.0.0 (tags/RELEASE_600/final 327031) (llvm/tags/RELEASE_600/final 327028)').should.equal(true);

        compiler.isCfgCompiler('g++ (GCC-Explorer-Build) 4.9.4').should.equal(true);
        compiler.isCfgCompiler('g++ (GCC-Explorer-Build) 8.0.1 20180223 (experimental)').should.equal(true);
        compiler.isCfgCompiler('g++ (GCC-Explorer-Build) 8.0.1 20180223 (experimental)').should.equal(true);
        compiler.isCfgCompiler('g++ (GCC) 4.1.2').should.equal(true);

        compiler.isCfgCompiler('foo-bar-g++ (GCC-Explorer-Build) 4.9.4').should.equal(true);
        compiler.isCfgCompiler('foo-bar-gcc (GCC-Explorer-Build) 4.9.4').should.equal(true);
        compiler.isCfgCompiler('foo-bar-gdc (GCC-Explorer-Build) 4.9.4').should.equal(true);

        compiler.isCfgCompiler('fake-for-test (Based on g++)').should.equal(false);

        compiler.isCfgCompiler('gdc (crosstool-NG 203be35 - 20160205-2.066.1-e95a735b97) 5.2.0').should.equal(true);
        compiler.isCfgCompiler('gdc (crosstool-NG hg+unknown-20131212.080758 - 20140430-2.064.2-404a037d26) 4.8.2').should.equal(true);
        compiler.isCfgCompiler('gdc (crosstool-NG crosstool-ng-1.20.0-232-gc746732 - 20150830-2.066.1-d0dd4a83de) 4.9.3').should.equal(true);

        compiler.isCfgCompiler('fake-for-test (Based on gdc)').should.equal(false);
    });
    it('should allow comments next to includes (Bug #874)', () => {
        should.equal(compiler.checkSource('#include <cmath> // std::(sin, cos, ...)'), null);
        const badSource = compiler.checkSource('#include </dev/null..> //Muehehehe');
        should.exist(badSource);
        badSource.should.equal('<stdin>:1:1: no absolute or relative includes please');
    });
});

describe('Compiler execution', function () {
    let ce, compiler, compilerNoExec;
    const executingCompilerInfo = {
        exe: null,
        remote: true,
        lang: languages['c++'].id,
        ldPath: [],
        supportsExecute: true,
        supportsBinary: true,
    };
    const noExecuteSupportCompilerInfo = {
        exe: null,
        remote: true,
        lang: languages['c++'].id,
        ldPath: [],
    };

    before(() => {
        ce = makeCompilationEnvironment({ languages });
        compiler = new BaseCompiler(executingCompilerInfo, ce);
        compilerNoExec = new BaseCompiler(noExecuteSupportCompilerInfo, ce);
    });

    afterEach(() => sinon.restore());

    function stubOutCallToExec(execStub, compiler, content, result, nthCall) {
        execStub.onCall(nthCall || 0).callsFake((compiler, args) => {
            const minusO = args.indexOf('-o');
            minusO.should.be.gte(0);
            const output = args[minusO + 1];
            // Maybe we should mock out the FS too; but that requires a lot more work.
            fs.writeFileSync(output, content);
            result.filenameTransform = x => x;
            return Promise.resolve(result);
        });
    }

    it('should compile', async () => {
        const execStub = sinon.stub(compiler, 'exec');
        stubOutCallToExec(execStub, compiler, 'This is the output file', {
            code: 0,
            okToCache: true,
            stdout: 'stdout',
            stderr: 'stderr',
        });
        const result = await compiler.compile(
            'source',
            'options',
            {},
            {},
            false,
            [],
            {},
            []);
        result.code.should.equal(0);
        result.compilationOptions.should.contain('options');
        result.compilationOptions.should.contain(result.inputFilename);
        result.okToCache.should.be.true;
        result.asm.should.deep.equal([{source: null, text: 'This is the output file', labels: []}]);
        result.stdout.should.deep.equal([{text: 'stdout'}]);
        result.stderr.should.deep.equal([{text: 'stderr'}]);
        result.popularArguments.should.deep.equal({});
        result.tools.should.deep.equal([]);
        execStub.called.should.be.true;
    });

    it('should handle compilation failures', async () => {
        const execStub = sinon.stub(compiler, 'exec');
        stubOutCallToExec(execStub, compiler, 'This is the output file', {
            code: 1,
            okToCache: true,
            stdout: '',
            stderr: 'oh noes',
        });
        const result = await compiler.compile(
            'source',
            'options',
            {},
            {},
            false,
            [],
            {},
            []);
        result.code.should.equal(1);
        result.asm.should.deep.equal([{labels: [], source: null, text: '<Compilation failed>'}]);
    });

    it('should cache results (when asked)', async () => {
        const ceMock = sinon.mock(ce);
        const fakeExecResults = {
            code: 0,
            okToCache: true,
            stdout: 'stdout',
            stderr: 'stderr',
        };
        const execStub = sinon.stub(compiler, 'exec');
        stubOutCallToExec(execStub, compiler, 'This is the output file', fakeExecResults);
        const source = 'Some cacheable source';
        const options = 'Some cacheable options';
        ceMock.expects('cachePut').withArgs(sinon.match({source, options}), sinon.match(fakeExecResults)).resolves();
        const uncachedResult = await compiler.compile(
            source,
            options,
            {},
            {},
            false,
            [],
            {},
            []);
        uncachedResult.code.should.equal(0);
        ceMock.verify();
    });

    it('should not cache results (when not asked)', async () => {
        const ceMock = sinon.mock(ce);
        const fakeExecResults = {
            code: 0,
            okToCache: false,
            stdout: 'stdout',
            stderr: 'stderr',
        };
        const execStub = sinon.stub(compiler, 'exec');
        stubOutCallToExec(execStub, compiler, 'This is the output file', fakeExecResults);
        ceMock.expects('cachePut').never();
        const source = 'Some cacheable source';
        const options = 'Some cacheable options';
        const uncachedResult = await compiler.compile(
            source,
            options,
            {},
            {},
            false,
            [],
            {},
            []);
        uncachedResult.code.should.equal(0);
        ceMock.verify();
    });

    it('should read from the cache (when asked)', async () => {
        const ceMock = sinon.mock(ce);
        const source = 'Some previously cached source';
        const options = 'Some previously cached options';
        ceMock.expects('cacheGet').withArgs(sinon.match({source, options})).resolves({code: 123});
        const cachedResult = await compiler.compile(
            source,
            options,
            {},
            {},
            false,
            [],
            {},
            []);
        cachedResult.code.should.equal(123);
        ceMock.verify();
    });

    it('should note read from the cache (when bypassed)', async () => {
        const ceMock = sinon.mock(ce);
        const fakeExecResults = {
            code: 0,
            okToCache: true,
            stdout: 'stdout',
            stderr: 'stderr',
        };
        const source = 'Some previously cached source';
        const options = 'Some previously cached options';
        ceMock.expects('cacheGet').never();
        const execStub = sinon.stub(compiler, 'exec');
        stubOutCallToExec(execStub, compiler, 'This is the output file', fakeExecResults);
        const uncachedResult = await compiler.compile(
            source,
            options,
            {},
            {},
            true,
            [],
            {},
            []);
        uncachedResult.code.should.equal(0);
        ceMock.verify();
    });

    it('should execute', async () => {
        const execMock = sinon.mock(exec);
        const execStub = sinon.stub(compiler, 'exec');
        stubOutCallToExec(execStub, compiler, 'This is the output asm file', {
            code: 0,
            okToCache: true,
            stdout: 'asm stdout',
            stderr: 'asm stderr',
        }, 0);
        stubOutCallToExec(execStub, compiler, 'This is the output binary file', {
            code: 0,
            okToCache: true,
            stdout: 'binary stdout',
            stderr: 'binary stderr',
        }, 1);
        execMock.expects('sandbox').withArgs(sinon.match.string, sinon.match.array, sinon.match.object).resolves({
            code: 0,
            stdout: 'exec stdout',
            stderr: 'exec stderr',
        });
        const result = await compiler.compile(
            'source',
            'options',
            {},
            {execute: true},
            false,
            [],
            {},
            []);
        result.code.should.equal(0);
        result.execResult.didExecute.should.be.true;
        result.stdout.should.deep.equal([{text: 'asm stdout'}]);
        result.execResult.stdout.should.deep.equal([{text: 'exec stdout'}]);
        result.execResult.buildResult.stdout.should.deep.equal([{text: 'binary stdout'}]);
        result.stderr.should.deep.equal([{text: 'asm stderr'}]);
        result.execResult.stderr.should.deep.equal([{text: 'exec stderr'}]);
        result.execResult.buildResult.stderr.should.deep.equal([{text: 'binary stderr'}]);
    });

    it('should not execute where not supported', async () => {
        const execMock = sinon.mock(exec);
        const execStub = sinon.stub(compilerNoExec, 'exec');
        stubOutCallToExec(execStub, compilerNoExec, 'This is the output asm file', {
            code: 0,
            okToCache: true,
            stdout: 'asm stdout',
            stderr: 'asm stderr',
        }, 0);
        stubOutCallToExec(execStub, compilerNoExec, 'This is the output binary file', {
            code: 0,
            okToCache: true,
            stdout: 'binary stdout',
            stderr: 'binary stderr',
        }, 1);
        execMock.expects('sandbox').withArgs(sinon.match.string, sinon.match.array, sinon.match.object).resolves({
            code: 0,
            stdout: 'exec stdout',
            stderr: 'exec stderr',
        });
        const result = await compilerNoExec.compile(
            'source',
            'options',
            {},
            {execute: true},
            false,
            [],
            {},
            []);
        result.code.should.equal(0);
        result.execResult.didExecute.should.be.false;
        result.stdout.should.deep.equal([{text: 'asm stdout'}]);
        result.execResult.stdout.should.deep.equal([]);
        result.execResult.buildResult.stdout.should.deep.equal([{text: 'binary stdout'}]);
        result.stderr.should.deep.equal([{text: 'asm stderr'}]);
        result.execResult.stderr.should.deep.equal([{text: 'Compiler does not support execution'}]);
        result.execResult.buildResult.stderr.should.deep.equal([{text: 'binary stderr'}]);
    });

    it('should demangle', async () => {
        const withDemangler = {...noExecuteSupportCompilerInfo, demangler: 'demangler-exe', demanglerType: 'cpp'};
        const compiler = new BaseCompiler(withDemangler, ce);
        const execStub = sinon.stub(compiler, 'exec');
        stubOutCallToExec(execStub, compiler, 'someMangledSymbol:\n', {
            code: 0,
            okToCache: true,
            stdout: 'stdout',
            stderr: 'stderr',
        });
        execStub.onCall(1).callsFake((demangler, args, options) => {
            demangler.should.equal('demangler-exe');
            options.input.should.equal('someMangledSymbol');
            return Promise.resolve({
                code: 0,
                filenameTransform: x => x,
                stdout: 'someDemangledSymbol\n',
                stderr: '',
            });
        });
        const result = await compiler.compile(
            'source',
            'options',
            {},
            {demangle: true},
            false,
            [],
            {},
            []);
        result.code.should.equal(0);
        result.asm.should.deep.equal([{source: null, labels: [], text: 'someDemangledSymbol:'}]);
        // TODO all with demangle: false
    });

    it('should run objdump properly', async () => {
        const withDemangler = {...noExecuteSupportCompilerInfo, objdumper: 'objdump-exe'};
        const compiler = new BaseCompiler(withDemangler, ce);
        const execStub = sinon.stub(compiler, 'exec');
        execStub.onCall(0).callsFake((objdumper, args, options) => {
            objdumper.should.equal('objdump-exe');
            args.should.deep.equal([
                '-d', 'output',
                '-l', '--insn-width=16',
                '-C', '-M', 'intel']);
            options.maxOutput.should.equal(123456);
            return Promise.resolve({
                code: 0,
                filenameTransform: x => x,
                stdout: 'the output',
                stderr: '',
            });
        });
        compiler.supportsObjdump().should.be.true;
        const result = await compiler.objdump(
            'output',
            {},
            123456,
            true,
            true);
        result.asm.should.deep.equal('the output');
    });

    it('should run process opt output', async () => {
        const test = `--- !Missed
Pass: inline
Name: NeverInline
DebugLoc: { File: example.cpp, Line: 4, Column: 21 }
Function: main
Args: []
...
`;
        const dirPath = await compiler.newTempDir();
        const optPath = path.join(dirPath, 'temp.out');
        await fs.writeFile(optPath, test);
        const a = await compiler.processOptOutput(optPath);
        a.should.deep.equal([{
            Args: [],
            DebugLoc: {Column: 21, File: 'example.cpp', Line: 4},
            Function: 'main',
            Name: 'NeverInline',
            Pass: 'inline',
            displayString: '',
            optType: 'Missed',
        }]);
    });
});
