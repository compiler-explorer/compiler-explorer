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

import {beforeAll, describe, expect, it} from 'vitest';

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

function makeCompiler(stdout?: string, stderr?: string, code?: number) {
    if (code === undefined) code = 0;
    const compiler = new FakeCompiler({lang: 'c++'}) as any;
    compiler.exec = () => Promise.resolve({code: code, stdout: stdout || '', stderr: stderr || ''});
    compiler.execCompilerCached = compiler.exec;
    compiler.possibleArguments = new CompilerArguments('g82');
    return compiler;
}

describe('option parser', () => {
    it('should do nothing for the base parser', async () => {
        const compiler = makeCompiler();
        await expect(BaseParser.parse(compiler)).resolves.toEqual(compiler);
    });
    it('should handle empty options', async () => {
        await expect(BaseParser.getOptions(makeCompiler(), '')).resolves.toEqual({});
    });
    it('should parse single-dash options', async () => {
        await expect(BaseParser.getOptions(makeCompiler('-foo\n'), '')).resolves.toEqual({
            '-foo': {
                description: '',
                timesused: 0,
            },
        });
    });
    it('should parse double-dash options', async () => {
        await expect(BaseParser.getOptions(makeCompiler('--foo\n'), '')).resolves.toEqual({
            '--foo': {
                description: '',
                timesused: 0,
            },
        });
    });
    it('should parse stderr options', async () => {
        await expect(BaseParser.getOptions(makeCompiler('', '--bar=monkey\n'), '')).resolves.toEqual({
            '--bar=monkey': {
                description: '',
                timesused: 0,
            },
        });
    });
    it('handles non-option text', async () => {
        await expect(BaseParser.getOptions(makeCompiler('-foo=123\nthis is a fish\n-badger=123'), '')).resolves.toEqual(
            {
                '-foo=123': {description: 'this is a fish', timesused: 0},
                '-badger=123': {description: '', timesused: 0},
            },
        );
    });
    it('should ignore if errors occur', async () => {
        await expect(BaseParser.getOptions(makeCompiler('--foo\n', '--bar\n', 1), '')).resolves.toEqual({});
    });
});

describe('gcc parser', () => {
    it('should handle empty options', async () => {
        const result = await GCCParser.parse(makeCompiler());
        expect(result.compiler).not.toHaveProperty('supportsGccDump');
        expect(result.compiler.options).toEqual('');
    });
    it('should handle options', async () => {
        const result = await GCCParser.parse(makeCompiler('-masm=intel\n-fdiagnostics-color=[blah]\n-fdump-tree-all'));
        expect(result.compiler.supportsGccDump).toBe(true);
        expect(result.compiler.supportsIntel).toBe(true);
        expect(result.compiler.intelAsm).toEqual('-masm=intel');
        expect(result.compiler.options).toEqual('-fdiagnostics-color=always');
    });
    it('should handle undefined options', async () => {
        const result = await GCCParser.parse(makeCompiler('-fdiagnostics-color=[blah]'));
        expect(result.compiler.options).toEqual('-fdiagnostics-color=always');
    });
});

describe('clang parser', () => {
    it('should handle empty options', async () => {
        const result = await ClangParser.parse(makeCompiler());
        expect(result.compiler.options).toEqual('');
    });
    it('should handle options', async () => {
        const result = await ClangParser.parse(
            makeCompiler('  -fno-crash-diagnostics\n  -fsave-optimization-record\n  -fcolor-diagnostics'),
        );
        expect(result.compiler.supportsOptOutput).toBe(true);
        expect(result.compiler.optArg).toEqual('-fsave-optimization-record');
        expect(result.compiler.options).toContain('-fcolor-diagnostics');
        expect(result.compiler.options).toContain('-fno-crash-diagnostics');
        expect(result.compiler.options).not.toContain('-fsave-optimization-record');
    });
});

describe('pascal parser', () => {
    it('should handle empty options', async () => {
        const result = await PascalParser.parse(makeCompiler());
        expect(result.compiler.options).toEqual('');
    });
});

describe('popular compiler arguments', () => {
    let compiler;

    beforeAll(() => {
        compiler = makeCompiler(
            '  -fsave-optimization-record\n  -x\n  -g\n  -fcolor-diagnostics\n  -O<number>  Optimization level\n  -std=<c++11,c++14,c++17z>',
        );
    });

    it('should return 5 arguments', async () => {
        const result = await ClangParser.parse(compiler);
        expect(result.possibleArguments.getPopularArguments()).toEqual({
            '-O<number>': {description: 'Optimization level', timesused: 0},
            '-fcolor-diagnostics': {description: '', timesused: 0},
            '-fsave-optimization-record': {description: '', timesused: 0},
            '-g': {description: '', timesused: 0},
            '-x': {description: '', timesused: 0},
        });
    });

    it('should return arguments except the ones excluded', async () => {
        const result = await ClangParser.parse(compiler);
        expect(result.possibleArguments.getPopularArguments(['-O3', '--hello'])).toEqual({
            '-fcolor-diagnostics': {description: '', timesused: 0},
            '-fsave-optimization-record': {description: '', timesused: 0},
            '-g': {description: '', timesused: 0},
            '-x': {description: '', timesused: 0},
            '-std=<c++11,c++14,c++17z>': {description: '', timesused: 0},
        });
    });

    it('should be able to exclude special params with assignments', async () => {
        const result = await ClangParser.parse(compiler);
        expect(result.possibleArguments.getPopularArguments(['-std=c++14', '-g', '--hello'])).toEqual({
            '-O<number>': {description: 'Optimization level', timesused: 0},
            '-fcolor-diagnostics': {description: '', timesused: 0},
            '-fsave-optimization-record': {description: '', timesused: 0},
            '-x': {description: '', timesused: 0},
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
        expect(stdvers).toEqual([
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
        expect(stdvers).toEqual([
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
        expect(actions).toEqual([
            {name: 'gen-attrs: Generate attributes', value: '--gen-attrs'},
            {name: 'print-detailed-records: Print full details...', value: '--print-detailed-records'},
            {name: 'gen-x86-mnemonic-tables: Generate X86...', value: '--gen-x86-mnemonic-tables'},
        ]);
    });
});
