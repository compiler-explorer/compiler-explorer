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

import fs from 'node:fs';
import path from 'node:path';

import {beforeAll, describe, expect, it} from 'vitest';

import {LLCParser} from '../lib/compilers/argument-parsers.js';
import {LLCCompiler} from '../lib/compilers/llc.js';
import {OptCompiler} from '../lib/compilers/opt.js';
import {makeCompilationEnvironment, newTempDir} from './utils.js';

const languages = {
    llvm: {id: 'llvm'},
};

let ce;

function createCompiler(compiler) {
    if (ce === undefined) {
        ce = makeCompilationEnvironment({languages});
    }

    const info = {
        exe: null,
        remote: true,
        lang: languages.llvm.id,
    };

    return new compiler(info, ce);
}

describe('LLVM IR Compiler', () => {
    let compiler;

    beforeAll(() => {
        compiler = createCompiler(LLCCompiler);
    });

    it('llc options for at&t assembly', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: false,
                    binary: false,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s']);
    });

    it('llc options for intel assembly', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: true,
                    binary: false,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-x86-asm-syntax=intel']);
    });

    it('llc options for at&t binary', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: false,
                    binary: true,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-filetype=obj']);
    });

    it('llc options for intel binary', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: true,
                    binary: true,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-filetype=obj']);
    });

    it('llc options for binary object', () => {
        expect(
            compiler.optionsForFilter(
                {
                    intel: true,
                    binaryObject: true,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-filetype=obj']);
    });

    it('llc takes its target triple through -mtriple', () => {
        expect(compiler.getTargetHintFromCompilerArgs(['-O2', '-mtriple=aarch64-linux-gnu'])).toEqual(
            'aarch64-linux-gnu',
        );
    });

    it('llc links for the host with a native toolchain', () => {
        expect(compiler.getLinkerArguments(undefined)).toEqual([]);
        expect(compiler.getLinkerArguments('/opt/gcc-14.2.0')).toEqual(['--gcc-toolchain=/opt/gcc-14.2.0']);
    });

    it('llc links for the cross toolchain target', () => {
        const toolchain = path.join(newTempDir(), 'aarch64-unknown-linux-gnu');
        const sysroot = path.join(toolchain, 'aarch64-unknown-linux-gnu', 'sysroot');
        fs.mkdirSync(sysroot, {recursive: true});

        expect(compiler.getLinkerArguments(toolchain)).toEqual([
            '--gcc-toolchain=' + toolchain,
            '--target=aarch64-unknown-linux-gnu',
            '--sysroot=' + sysroot,
        ]);
    });

    it('llc lists its registered targets', () => {
        const parser = new LLCParser(compiler);
        const lines = [
            'LLVM (http://llvm.org/):',
            '  LLVM version 21.1.0',
            '  Default target: x86_64-unknown-linux-gnu',
            '',
            '  Registered Targets:',
            '    aarch64     - AArch64 (little endian)',
            '    riscv64     - 64-bit RISC-V',
            '    x86-64      - 64-bit X86: EM64T and AMD64',
        ];
        expect(parser.extractPossibleTargets(lines)).toEqual(['aarch64', 'riscv64', 'x86-64']);
    });

    it('opt options', () => {
        const compiler = createCompiler(OptCompiler);

        expect(
            compiler.optionsForFilter(
                {
                    intel: false,
                    binary: false,
                },
                'output.s',
            ),
        ).toEqual(['-o', 'output.s', '-S']);
    });
});
