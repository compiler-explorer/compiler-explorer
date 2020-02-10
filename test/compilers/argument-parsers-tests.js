// Copyright (c) 2017, Matt Godbolt
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
    FakeCompiler = require('../../lib/compilers/fake-for-test'),
    CompilationEnvironment = require('../../lib/compilation-env'),
    chaiAsPromised = require("chai-as-promised"),
    parsers = require('../../lib/compilers/argument-parsers'),
    CompilerArguments = require('../../lib/compiler-arguments'),
    properties = require('../../lib/properties');
chai.use(chaiAsPromised);
const should = chai.should();

const languages = {
    'c++': {id: 'c++'}
};

const compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));

function makeCompiler(stdout, stderr, code) {
    if (code === undefined) code = 0;
    const env = new CompilationEnvironment(compilerProps);
    const compiler = new FakeCompiler({lang: languages['c++'].id, remote: true}, env);
    compiler.exec = () => Promise.resolve({code: code, stdout: stdout || "", stderr: stderr || ""});
    compiler.execCompilerCached = compiler.exec;
    compiler.possibleArguments = new CompilerArguments("g82");
    return compiler;
}

describe('option parser', () => {
    it('should do nothing for the base parser', () => {
        const compiler = makeCompiler();
        return parsers.Base.parse(compiler).should.deep.equals(compiler);
    });
    it('should handle empty options', () => {
        return parsers.Base.getOptions(makeCompiler()).should.eventually.deep.equals({});
    });
    it('should parse single-dash options', () => {
        return parsers.Base.getOptions(makeCompiler("-foo\n")).should.eventually.deep.equals({
            '-foo': {
                "description": "",
                "timesused": 0
            }
        });
    });
    it('should parse double-dash options', () => {
        return parsers.Base.getOptions(makeCompiler("--foo\n")).should.eventually.deep.equals({
            '--foo': {
                "description": "",
                "timesused": 0
            }
        });
    });
    it('should parse stderr options', () => {
        return parsers.Base.getOptions(makeCompiler("", "--bar=monkey\n")).should.eventually.deep.equals({
            '--bar=monkey': {
                "description": "",
                "timesused": 0
            }
        });
    });
    it('handles non-option text', () => {
        return parsers.Base.getOptions(makeCompiler("-foo=123\nthis is a fish\n-badger=123")).should.eventually.deep.equals(
            {
                '-foo=123': {"description": "this is a fish", "timesused": 0},
                '-badger=123': {"description": "", "timesused": 0}
            });
    });
    it('should ignore if errors occur', () => {
        return parsers.Base.getOptions(makeCompiler("--foo\n", "--bar\n", 1)).should.eventually.deep.equals({});
    });
});

describe('gcc parser', () => {
    it('should handle empty options', () => {
        return parsers.GCC.parse(makeCompiler()).should.eventually.satisfy(result => {
            return Promise.all([
                should.not.equal(result.compiler.supportsGccDump, true),
                result.compiler.options.should.equals('')
            ]);
        });
    });
    it('should handle options', () => {
        return parsers.GCC.parse(makeCompiler("-masm=intel\n-fdiagnostics-color=[blah]\n-fdump-tree-all"))
            .should.eventually.satisfy(result => {
                return Promise.all([
                    result.compiler.supportsGccDump.should.equals(true),
                    result.compiler.supportsIntel.should.equals(true),
                    result.compiler.intelAsm.should.equals('-masm=intel'),
                    result.compiler.options.should.equals('-fdiagnostics-color=always')
                ]);
            });
    });
    it('should handle undefined options', () => {
        return parsers.GCC.parse(makeCompiler("-fdiagnostics-color=[blah]")).should.eventually.satisfy(result => {
            return Promise.all([
                result.compiler.options.should.equals('-fdiagnostics-color=always')
            ]);
        });
    });
});

describe('clang parser', () => {
    it('should handle empty options', () => {
        return parsers.Clang.parse(makeCompiler()).should.eventually.satisfy(result => {
            return Promise.all([
                result.compiler.options.should.equals('')
            ]);
        });
    });
    it('should handle options', () => {
        return parsers.Clang.parse(makeCompiler("-fno-crash-diagnostics\n-fsave-optimization-record\n-fcolor-diagnostics"))
            .should.eventually.satisfy(result => {
                return Promise.all([
                    result.compiler.supportsOptOutput.should.equals(true),
                    result.compiler.optArg.should.equals('-fsave-optimization-record'),

                    result.compiler.options.should.include("-fcolor-diagnostics"),
                    result.compiler.options.should.include("-fno-crash-diagnostics"),
                    result.compiler.options.should.not.include("-fsave-optimization-record")
                ]);
            });
    });
});

describe('pascal parser', () => {
    it('should handle empty options', () => {
        return parsers.Pascal.parse(makeCompiler()).should.eventually.satisfy(result => {
            return Promise.all([
                result.compiler.options.should.equals('')
            ]);
        });
    });
});

describe('popular compiler arguments', () => {
    let compiler = makeCompiler("-fsave-optimization-record\n-x\n-g\n-fcolor-diagnostics\n-O<number> optimization level\n-std=<c++11,c++14,c++17z>");

    it('should return 5 arguments', () => {
        return parsers.Clang.parse(compiler).then(compiler => {
            return compiler.should.satisfy(compiler => {
                return Promise.all([
                    compiler.possibleArguments.getPopularArguments().should.deep.equal({
                        "-O<number>": {description: "optimization level", timesused: 0},
                        "-fcolor-diagnostics": {"description": "", "timesused": 0},
                        "-fsave-optimization-record": {"description": "", "timesused": 0},
                        "-g": {"description": "", "timesused": 0},
                        "-x": {"description": "", "timesused": 0}
                    })
                ]);
            });
        });
    });

    it('should return arguments except the ones excluded', () => {
        return parsers.Clang.parse(compiler).then(compiler => {
            return compiler.should.satisfy(compiler => {
                return Promise.all([
                    compiler.possibleArguments.getPopularArguments(["-O3", "--hello"]).should.deep.equal({
                        "-fcolor-diagnostics": {"description": "", "timesused": 0},
                        "-fsave-optimization-record": {"description": "", "timesused": 0},
                        "-g": {"description": "", "timesused": 0},
                        "-x": {"description": "", "timesused": 0},
                        "-std=<c++11,c++14,c++17z>": {"description": "", "timesused": 0}
                    })
                ]);
            });
        });
    });

    it('should be able to exclude special params with assignments', () => {
        return parsers.Clang.parse(compiler).then(compiler => {
            return compiler.should.satisfy(compiler => {
                return Promise.all([
                    compiler.possibleArguments.getPopularArguments(["-std=c++14", "-g", "--hello"]).should.deep.equal({
                        "-O<number>": {description: "optimization level", timesused: 0},
                        "-fcolor-diagnostics": {"description": "", "timesused": 0},
                        "-fsave-optimization-record": {"description": "", "timesused": 0},
                        "-x": {"description": "", "timesused": 0}
                    })
                ]);
            });
        });
    });
});
