// Copyright (c) 2012-2018, Matt Godbolt
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

const chai = require('chai'),
    chaiAsPromised = require("chai-as-promised"),
    parsers = require('../../lib/compilers/argument-parsers');

chai.use(chaiAsPromised);
chai.should();

function makeCompiler(stdout, stderr, code) {
    if (code === undefined) code = 0;
    return {
        exec: () => {
            return Promise.resolve({code: code, stdout: stdout || "", stderr: stderr || ""});
        },
        compiler: {
            options: ''
        }
    };
}

describe('option parser', () => {
    it('should handle empty options', () => {
        return parsers.getOptions(makeCompiler()).should.eventually.deep.equals({});
    });
    it('should parse single-dash options', () => {
        return parsers.getOptions(makeCompiler("-foo\n")).should.eventually.deep.equals({'-foo': true});
    });
    it('should parse double-dash options', () => {
        return parsers.getOptions(makeCompiler("--foo\n")).should.eventually.deep.equals({'--foo': true});
    });
    it('should parse stderr options', () => {
        return parsers.getOptions(makeCompiler("", "--bar=monkey\n")).should.eventually.deep.equals({'--bar': true});
    });
    it('handles non-option text', () => {
        return parsers.getOptions(makeCompiler("-foo=123\nthis is a fish\n-badger=123")).should.eventually.deep.equals(
            {'-foo': true, '-badger': true});
    });
    it('should ignore if errors occur', () => {
        return parsers.getOptions(makeCompiler("--foo\n", "--bar\n", 1)).should.eventually.deep.equals({});
    });
});

describe('gcc parser', () => {
    it('should handle empty options', () => {
        return parsers.gcc(makeCompiler()).should.eventually.satisfy(result => {
            return result.compiler.should.deep.equals({
                supportsGccDump: true,
                options: ''
            });
        });
    });
    it('should handle options', () => {
        return parsers.gcc(makeCompiler("-masm=intel\n-fdiagnostics-color=[blah]"))
            .should.eventually.satisfy(result => {
                return result.compiler.should.deep.equals({
                    supportsGccDump: true,
                    supportsIntel: true,
                    intelAsm: '-masm=intel',
                    options: '-fdiagnostics-color=always'
                });
            });
    });
});

describe('clang parser', () => {
    it('should handle empty options', () => {
        return parsers.clang(makeCompiler()).should.eventually.satisfy(result => {
            return result.compiler.should.deep.equals({
                options: ''
            });
        });
    });
    it('should handle options', () => {
        return parsers.clang(makeCompiler("-fsave-optimization-record\n-fcolor-diagnostics"))
            .should.eventually.satisfy(result => {
                return result.compiler.should.deep.equals({
                    supportsOptOutput: true,
                    optArg: '-fsave-optimization-record',
                    options: '-fcolor-diagnostics'
                });
            });
    });
});
