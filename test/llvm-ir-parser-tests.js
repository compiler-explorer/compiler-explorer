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
const LlvmIrParser = require("../lib/llvm-ir").LlvmIrParser;
const properties = require('../lib/properties');

chai.use(chaiAsPromised);
chai.should();
const expect = chai.expect;

const languages = {
    'c++': {id: 'c++'}
};

let compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));
compilerProps = compilerProps.get.bind(compilerProps);

describe('llvm-ir getSourceLineNumber', function () {
    const llvmIrParser = new LlvmIrParser(compilerProps);
    const debugInfo = {
        '!10': { line: 10 },
        '!20': { line: 20, scope: '!10' },
        '!11': { scope: '!10' },
        '!12': { line: 0, scope: '!10' },
        '!14': { },
        '!15': { scope: '!14' },
        '!16': { scope: '!42' }
    }

    it('should return a line number', function () {
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!10')).to.equal(10);
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!20')).to.equal(20);
    });

    it('should return the line number of its parent scope', function () {
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!11')).to.equal(10);
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!12')).to.equal(10);
    });

    it('should return null on non-existend node', function () {
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!16')).to.equal(null);
    });

    it('should return null if no higher scope has a line', function () {
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!14')).to.equal(null);
        expect(llvmIrParser.getSourceLineNumber(debugInfo, '!15')).to.equal(null);
    });
});

describe('llvm-ir getFileName', function () {
    const llvmIrParser = new LlvmIrParser(compilerProps);
    const debugInfo = {
        '!10': { filename: "/test.cpp" },
        '!20': { filename: "/example.cpp" },
        '!11': { file: '!10' },
        '!21': { file: '!20' },
        '!12': { scope: '!11' },
        '!13': { scope: '!12' }
    }

    it('should return a filename', function () {
        expect(llvmIrParser.getFileName(debugInfo, '!10')).to.equal("/test.cpp");
        expect(llvmIrParser.getFileName(debugInfo, '!11')).to.equal("/test.cpp");
    });

    it('should return the filename of its parent scope', function () {
        expect(llvmIrParser.getFileName(debugInfo, '!12')).to.equal("/test.cpp");
        expect(llvmIrParser.getFileName(debugInfo, '!13')).to.equal("/test.cpp");
    });

    it('should return null on non-existend node', function () {
        expect(llvmIrParser.getFileName(debugInfo, '!42')).to.equal(null);
    });

    it('should return not return source filename', function () {
        expect(llvmIrParser.getFileName(debugInfo, '!20')).to.equal(null);
        expect(llvmIrParser.getFileName(debugInfo, '!21')).to.equal(null);
    });
});
