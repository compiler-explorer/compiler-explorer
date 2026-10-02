// Copyright (c) 2025, Compiler Explorer Authors
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

import {beforeAll, describe, expect, it, vi} from 'vitest';

import type {CompilationEnvironment} from '../lib/compilation-env.js';
import {BaseParser} from '../lib/compilers/argument-parsers.js';
import {decode_symbols, NumbaCompiler} from '../lib/compilers/numba.js';
import type {AsmResultSource} from '../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult} from '../types/compilation/compilation.interfaces.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

describe('Numba', () => {
    let ce: CompilationEnvironment;
    const languages = {
        numba: {id: 'numba' as LanguageKey},
    };
    const info = {
        exe: 'none',
        lang: languages.numba.id,
    };
    const filters = {
        binary: false,
        execute: false,
        demangle: true,
        intel: true,
        commentOnly: true,
        directives: true,
        labels: true,
        optOutput: false,
        libraryCode: false,
        trim: true,
        binaryObject: false,
        debugCalls: false,
    };
    const options: string[] = [];

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('should quack like a numba compiler', () => {
        const compiler = new NumbaCompiler(makeFakeCompilerInfo(info), ce);
        expect(compiler.getArgumentParserClass()).toBe(BaseParser);
        expect(NumbaCompiler.key).toEqual(languages.numba.id);
    });

    it('should give good wrapper script arguments', () => {
        const compiler = new NumbaCompiler(makeFakeCompilerInfo(info), ce);
        const outputFilename = 'test.log';
        const options = compiler.optionsForFilter({}, outputFilename);
        expect(options[0]).toEqual('-I');
        expect(options[1]).toContain('numba_wrapper.py');
        const i_outputfile = options.indexOf('--outputfile');
        expect(i_outputfile).not.toEqual(-1);
        expect(options[i_outputfile + 1]).toEqual(outputFilename);
        expect(options.at(-1)!).toEqual('--inputfile');
    });

    it('processing should filter and add line numbers', async () => {
        const compiler = new NumbaCompiler(makeFakeCompilerInfo(info), ce);
        const asm =
            ' .text;123\n' +
            ' .file   "<string>";123\n' +
            ' .globl  _ZNmangledEdd;123\n' +
            ' .p2align        4, 0x90;123\n' +
            ' .type   _ZNmangledEdd,@function;123\n' +
            '_ZNmangledEdd:;123\n' +
            '  pushq   %rbx;123\n' +
            '  movq    %rdi, %rbx;123\n' +
            '  movabsq $pow, %rax;123\n' +
            '  callq   *%rax;123\n' +
            '  vmovsd  %xmm0, (%rbx);123\n' +
            '  xorl    %eax, %eax;123\n' +
            '  popq    %rbx;123\n' +
            '  retq;123\n' +
            '.Lfunc_end0:;123\n' +
            ' .size   _ZNmangledEdd, .Lfunc_end0-_ZNmangledEdd;123\n';
        const processed = await compiler.processAsm({asm}, filters, options);
        expect(processed.asm.map(item => item.text)).toEqual([
            '_ZNmangledEdd:',
            '  pushq %rbx',
            '  movq %rdi, %rbx',
            '  movabsq $pow, %rax',
            '  callq *%rax',
            '  vmovsd %xmm0, (%rbx)',
            '  xorl %eax, %eax',
            '  popq %rbx',
            '  retq',
        ]);
        expect(processed.asm[0].source).toBeNull();
        for (const item of processed.asm.slice(1)) {
            expect(item.source).not.toBeNull();
            expect(item.source).not.toBeUndefined();
            const source = item.source as AsmResultSource;
            expect(source.line).not.toBeNull();
            expect(source.line).not.toBeUndefined();
            const line = source.line as number;
            expect(line).toEqual(123);
        }
    });

    it('should demangle special strings', async () => {
        const compiler = new NumbaCompiler(makeFakeCompilerInfo(info), ce);
        const asm = [
            // These text strings were pre-processed by the default demangler.
            {
                text:
                    'example::factory::_3clocals_3e::power[abi:v1][abi:c8tJTIcFKzyF2ILShI4CrgQElQb6HczSBAA_3d]' +
                    '(double, double):',
            },
            {
                text:
                    'example::xorshift64::next(' +
                    'uint64_20generator_28func_3d_3cfunction_20xorshift64_20at_200x7fd5948c18a0_3e_2c_20args_3d' +
                    '_28int64_2c_29_2c_20has_finalizer_3dTrue_29):',
            },
            {
                text: 'example::square(Array<double, 1, C, mutable, aligned>):',
            },
        ];
        const processed = await compiler.postProcessAsm({asm}, filters);
        expect(processed.asm.map(item => item.text)).toEqual([
            'example::factory::<locals>::power(double, double):',
            'example::xorshift64::next(uint64 generator(func=<function xorshift64 at 0x7fd5948c18a0>, args=(int64,), ' +
                'has_finalizer=True)):',
            'example::square(Array<double, 1, C, mutable, aligned>):',
        ]);
    });

    function fakeCompileResult(overrides: Partial<CompilationResult> = {}): CompilationResult {
        return {
            code: 0,
            timedOut: false,
            stdout: [],
            stderr: [],
            ...overrides,
        };
    }

    it('should dump passes on a separate output file and report failures', async () => {
        const compiler = new NumbaCompiler(makeFakeCompilerInfo(info), ce);
        const options = ['-I', 'numba_wrapper.py', '--outputfile', '/tmp/asm.s', '--inputfile', '/tmp/example.py'];
        const run = vi.spyOn(compiler, 'runCompiler').mockResolvedValue(
            fakeCompileResult({
                stdout: [
                    {text: '------<dynamic>.square: nopython: AFTER translate_bytecode------'},
                    {text: 'label 0:'},
                ],
            }),
        );

        const parsed = await compiler.generateOptPipeline('/tmp/example.py', options, filters, {});

        const [, args, input, execOptions] = run.mock.calls[0];
        expect(args[args.indexOf('--outputfile') + 1]).not.toBe('/tmp/asm.s');
        expect(args[args.indexOf('--inputfile') + 1]).toBe('/tmp/example.py');
        expect(input).toBe('/tmp/example.py');
        expect(options[options.indexOf('--outputfile') + 1]).toBe('/tmp/asm.s');
        expect(execOptions.env.NUMBA_DEBUG_PRINT_AFTER).toBe('all');
        expect(parsed?.results['<dynamic>.square'][0].name).toBe('translate_bytecode');

        run.mockResolvedValue(fakeCompileResult({code: 255, stderr: [{text: 'File "<source>", line 4'}]}));
        const failed = await compiler.generateOptPipeline('/tmp/example.py', options, filters, {});
        expect(failed?.error).toContain('File "<source>", line 4');
        expect(failed?.results).toEqual({});

        run.mockResolvedValue(fakeCompileResult({truncated: true, stdout: [{text: 'label 0:'}]}));
        const truncated = await compiler.generateOptPipeline('/tmp/example.py', options, filters, {});
        expect(truncated?.error).toBe('Exceeded max output limit');
        expect(truncated?.results).toEqual({});
    });

    it('should invert the encoding accurately', () => {
        expect(decode_symbols('plain_name123')).toEqual('plain_name123');
        expect(decode_symbols('_3clocals_3e')).toEqual('<locals>');
        const all_ascii = [...Array.from({length: 128}).keys()].map(i => String.fromCodePoint(i)).join('');
        const encode = (c: string) => '_' + c.codePointAt(0)!.toString(16).padStart(2, '0');
        const all_ascii_encoded = all_ascii.replaceAll(/[^\d_a-z]/g, encode);
        expect(decode_symbols(all_ascii_encoded)).toEqual(all_ascii);
    });
});
