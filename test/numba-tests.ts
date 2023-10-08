// Copyright (c) 2023, Compiler Explorer Authors
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

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {BaseParser} from '../lib/compilers/argument-parsers.js';
import {demangle_symbols, NumbaCompiler} from '../lib/compilers/numba.js';
import {LanguageKey} from '../types/languages.interfaces.js';

import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

describe('Numba', () => {
    let ce: CompilationEnvironment;
    const languages = {
        numba: {id: 'numba' as LanguageKey},
    };
    const info = {
        exe: '/dev/null',
        remote: {
            target: 'foo',
            path: 'bar',
        },
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
    const options = [];

    before(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('should quack like a Numba compiler', () => {
        const compiler = new NumbaCompiler(makeFakeCompilerInfo(info), ce);
        compiler.getArgumentParser().should.equal(BaseParser);
        NumbaCompiler.key.should.equal(languages.numba.id);
    });

    it('should give good wrapper script arguments', () => {
        const compiler = new NumbaCompiler(makeFakeCompilerInfo(info), ce);
        const outputFilename = 'test.log';
        const options = compiler.optionsForFilter({}, outputFilename);
        options[0].should.equal('-I');
        options[1].should.contain('numba_wrapper.py');
        const i_outputfile = options.indexOf('--outputfile');
        i_outputfile.should.not.equal(-1);
        options[i_outputfile + 1].should.equal(outputFilename);
        options.at(-1)!.should.equal('--inputfile');
    });

    it('processeing should filter and add line numbers', async () => {
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
        processed.asm
            .map(item => item.text)
            .should.deep.equal([
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
        processed.asm[0].source?.should.be.null;
        for (const item of processed.asm.slice(1)) {
            item.source.line.should.equal(123);
        }
    });

    it('should demangle special strings', async () => {
        const compiler = new NumbaCompiler(makeFakeCompilerInfo(info), ce);
        const asm = [
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
        ];
        const processed = await compiler.postProcessAsm({asm}, filters);
        processed.asm
            .map(item => item.text)
            .should.deep.equal([
                'example::factory::<locals>::power(double, double):',
                'example::xorshift64::next(uint64 generator(func=<function xorshift64 at 0x7fd5948c18a0>, args=(int64,), ' +
                    'has_finalizer=True)):',
            ]);
    });

    it('should invert the encoding with demangle_symbols', () => {
        demangle_symbols('plain_name123').should.equal('plain_name123');
        demangle_symbols('_3clocals_3e').should.equal('<locals>');
        const all_ascii = [...Array.from({length: 128}).keys()].map(i => String.fromCodePoint(i)).join('');
        const encode = (c: string) => '_' + c.codePointAt(0)!.toString(16).padStart(2, '0');
        const all_ascii_encoded = all_ascii.replaceAll(/[^\d_a-z]/g, encode);
        demangle_symbols(all_ascii_encoded).should.equal(all_ascii);
    });
});
