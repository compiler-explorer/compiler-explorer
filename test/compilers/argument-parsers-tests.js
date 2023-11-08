// Copyright (c) 2017, Compiler Explorer Authors
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

import {CompilerArguments} from '../../lib/compiler-arguments.js';
import {
    BaseParser,
    ClangParser,
    GCCParser,
    ICCParser,
    PascalParser,
    TableGenParser,
    VCParser,
} from '../../lib/compilers/argument-parsers.js';
import {FakeCompiler} from '../../lib/compilers/fake-for-test.js';
import {makeCompilationEnvironment, should} from '../utils.js';

const languages = {
    'c++': {id: 'c++'},
};

let env;

function makeCompiler(stdout, stderr, code) {
    if (env === undefined) {
        env = makeCompilationEnvironment({languages});
    }

    if (code === undefined) code = 0;
    const compiler = new FakeCompiler({lang: languages['c++'].id, remote: true}, env);
    compiler.exec = () => Promise.resolve({code: code, stdout: stdout || '', stderr: stderr || ''});
    compiler.execCompilerCached = compiler.exec;
    compiler.possibleArguments = new CompilerArguments('g82');
    return compiler;
}

describe('option parser', () => {
    it('should do nothing for the base parser', () => {
        const compiler = makeCompiler();
        return BaseParser.parse(compiler).should.deep.equals(compiler);
    });
    it('should handle empty options', () => {
        return BaseParser.getOptions(makeCompiler()).should.eventually.deep.equals({});
    });
    it('should parse single-dash options', () => {
        return BaseParser.getOptions(makeCompiler('-foo\n')).should.eventually.deep.equals({
            '-foo': {
                description: '',
                timesused: 0,
            },
        });
    });
    it('should parse double-dash options', () => {
        return BaseParser.getOptions(makeCompiler('--foo\n')).should.eventually.deep.equals({
            '--foo': {
                description: '',
                timesused: 0,
            },
        });
    });
    it('should parse stderr options', () => {
        return BaseParser.getOptions(makeCompiler('', '--bar=monkey\n')).should.eventually.deep.equals({
            '--bar=monkey': {
                description: '',
                timesused: 0,
            },
        });
    });
    it('handles non-option text', () => {
        return BaseParser.getOptions(
            makeCompiler('-foo=123\nthis is a fish\n-badger=123'),
        ).should.eventually.deep.equals({
            '-foo=123': {description: 'this is a fish', timesused: 0},
            '-badger=123': {description: '', timesused: 0},
        });
    });
    it('should ignore if errors occur', () => {
        return BaseParser.getOptions(makeCompiler('--foo\n', '--bar\n', 1)).should.eventually.deep.equals({});
    });
});

describe('gcc parser', () => {
    it('should handle empty options', async () => {
        const result = await GCCParser.parse(makeCompiler());
        should.not.exist(result.compiler.supportsGccDump);
        result.compiler.options.should.equals('');
    });
    it('should handle options', () => {
        return GCCParser.parse(
            makeCompiler('-masm=intel\n-fdiagnostics-color=[blah]\n-fdump-tree-all'),
        ).should.eventually.satisfy(result => {
            return Promise.all([
                result.compiler.supportsGccDump.should.equals(true),
                result.compiler.supportsIntel.should.equals(true),
                result.compiler.intelAsm.should.equals('-masm=intel'),
                result.compiler.options.should.equals('-fdiagnostics-color=always'),
            ]);
        });
    });
    it('should handle undefined options', () => {
        return GCCParser.parse(makeCompiler('-fdiagnostics-color=[blah]')).should.eventually.satisfy(result => {
            return Promise.all([result.compiler.options.should.equals('-fdiagnostics-color=always')]);
        });
    });
});

describe('clang parser', () => {
    it('should handle empty options', () => {
        return ClangParser.parse(makeCompiler()).should.eventually.satisfy(result => {
            return Promise.all([result.compiler.options.should.equals('')]);
        });
    });
    it('should handle options', () => {
        return ClangParser.parse(
            makeCompiler('  -fno-crash-diagnostics\n  -fsave-optimization-record\n  -fcolor-diagnostics'),
        ).should.eventually.satisfy(result => {
            return Promise.all([
                result.compiler.supportsOptOutput.should.equals(true),
                result.compiler.optArg.should.equals('-fsave-optimization-record'),

                result.compiler.options.should.include('-fcolor-diagnostics'),
                result.compiler.options.should.include('-fno-crash-diagnostics'),
                result.compiler.options.should.not.include('-fsave-optimization-record'),
            ]);
        });
    });
});

describe('pascal parser', () => {
    it('should handle empty options', () => {
        return PascalParser.parse(makeCompiler()).should.eventually.satisfy(result => {
            return Promise.all([result.compiler.options.should.equals('')]);
        });
    });
});

describe('popular compiler arguments', () => {
    let compiler;

    before(() => {
        compiler = makeCompiler(
            '  -fsave-optimization-record\n  -x\n  -g\n  -fcolor-diagnostics\n  -O<number>  Optimization level\n  -std=<c++11,c++14,c++17z>',
        );
    });

    it('should return 5 arguments', () => {
        return ClangParser.parse(compiler).then(compiler => {
            return compiler.should.satisfy(compiler => {
                return Promise.all([
                    compiler.possibleArguments.getPopularArguments().should.deep.equal({
                        '-O<number>': {description: 'Optimization level', timesused: 0},
                        '-fcolor-diagnostics': {description: '', timesused: 0},
                        '-fsave-optimization-record': {description: '', timesused: 0},
                        '-g': {description: '', timesused: 0},
                        '-x': {description: '', timesused: 0},
                    }),
                ]);
            });
        });
    });

    it('should return arguments except the ones excluded', () => {
        return ClangParser.parse(compiler).then(compiler => {
            return compiler.should.satisfy(compiler => {
                return Promise.all([
                    compiler.possibleArguments.getPopularArguments(['-O3', '--hello']).should.deep.equal({
                        '-fcolor-diagnostics': {description: '', timesused: 0},
                        '-fsave-optimization-record': {description: '', timesused: 0},
                        '-g': {description: '', timesused: 0},
                        '-x': {description: '', timesused: 0},
                        '-std=<c++11,c++14,c++17z>': {description: '', timesused: 0},
                    }),
                ]);
            });
        });
    });

    it('should be able to exclude special params with assignments', () => {
        return ClangParser.parse(compiler).then(compiler => {
            return compiler.should.satisfy(compiler => {
                return Promise.all([
                    compiler.possibleArguments.getPopularArguments(['-std=c++14', '-g', '--hello']).should.deep.equal({
                        '-O<number>': {description: 'Optimization level', timesused: 0},
                        '-fcolor-diagnostics': {description: '', timesused: 0},
                        '-fsave-optimization-record': {description: '', timesused: 0},
                        '-x': {description: '', timesused: 0},
                    }),
                ]);
            });
        });
    });
});

describe('VC argument parser', () => {
    it('Should extract stdversions', () => {
        const lines = [
            '   /helloWorld',
            '   /std:<c++14|c++17|c++20|c++latest> C++ standard version',
            '         c++14 - ISO/IEC 14882:2014 (default)',
            '         c++17 - ISO/IEC 14882:2017',
            '         c++20 - ISO/IEC 14882:2020',
            '         c++latest - latest draft standard (feature set subject to change)',
            '   /something:<else> Something Else',
            '   /etc Etcetera',
        ];
        const stdvers = VCParser.extractPossibleStdvers(lines);
        stdvers.should.deep.equal([
            {
                name: 'c++14: ISO/IEC 14882:2014 (default)',
                value: 'c++14',
            },
            {
                name: 'c++17: ISO/IEC 14882:2017',
                value: 'c++17',
            },
            {
                name: 'c++20: ISO/IEC 14882:2020',
                value: 'c++20',
            },
            {
                name: 'c++latest: latest draft standard (feature set subject to change)',
                value: 'c++latest',
            },
        ]);
    });
});

describe('ICC argument parser', () => {
    it('Should extract stdversions', () => {
        const lines = [
            '-test',
            '-std=<std>',
            '          enable language support for <std>, as described below',
            '            c99   conforms to ISO/IEC 9899:1999 standard for C programs',
            '            c++11 enables C++11 support for C++ programs',
            '            gnu++98 conforms to 1998 ISO C++ standard plus GNU extensions',
            '-etc',
        ];
        const stdvers = ICCParser.extractPossibleStdvers(lines);
        stdvers.should.deep.equal([
            {
                name: 'c99: conforms to ISO/IEC 9899:1999 standard for C programs',
                value: 'c99',
            },
            {
                name: 'c++11: enables C++11 support for C++ programs',
                value: 'c++11',
            },
            {
                name: 'gnu++98: conforms to 1998 ISO C++ standard plus GNU extensions',
                value: 'gnu++98',
            },
        ]);
    });
});

describe('TableGen argument parser', () => {
    it('Should extract actions', () => {
        const lines = [
            'USAGE: llvm-tblgen [options] <input file>',
            '',
            'OPTIONS:',
            '',
            'General options:',
            '',
            '  -D <macro name>                     - Name of the macro...',
            '  Action to perform:',
            '      --gen-attrs                        - Generate attributes',
            '      --print-detailed-records           - Print full details...',
            '      --gen-x86-mnemonic-tables          - Generate X86...',
            '  --no-warn-on-unused-template-args   - Disable...',
        ];
        const actions = TableGenParser.extractPossibleActions(lines);
        actions.should.deep.equal([
            {name: 'gen-attrs: Generate attributes', value: '--gen-attrs'},
            {name: 'print-detailed-records: Print full details...', value: '--print-detailed-records'},
            {name: 'gen-x86-mnemonic-tables: Generate X86...', value: '--gen-x86-mnemonic-tables'},
        ]);
    });
});
