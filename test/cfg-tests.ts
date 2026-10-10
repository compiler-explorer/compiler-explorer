// Copyright (c) 2017, Najjar Chedy
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

import * as fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';

import {describe, expect, it} from 'vitest';

import {generateStructure} from '../lib/cfg/cfg.js';
import {InstructionType, RiscvInstructionSetInfo} from '../lib/cfg/instruction-sets/index.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {makeFakeCompilerInfo, resolvePathFromTestRoot} from './utils.js';

async function DoCfgTest(cfgArg: string, filename: string, isLlvmIr = false, compilerInfo?: CompilerInfo) {
    const contents = JSON.parse(await fs.readFile(filename, 'utf8'));
    if (!compilerInfo) {
        compilerInfo = makeFakeCompilerInfo({
            compilerType: '',
            version: cfgArg,
        });
    }
    const structure = await generateStructure(compilerInfo, contents.asm, isLlvmIr);
    expect(structure).toEqual(contents.cfg);
}

describe('Cfg test cases', () => {
    const testcasespath = resolvePathFromTestRoot('cfg-cases');

    // For backwards compatability reasons, we have a sync readdir here. For details, see
    // the git blame of this file.
    // TODO: Consider replacing with https://github.com/vitest-dev/vitest/issues/703
    const files = fsSync.readdirSync(testcasespath);

    describe('base', () => {
        for (const filename of files.filter(x => x.includes('cfg-base'))) {
            it(filename, async () => {
                await DoCfgTest('', path.join(testcasespath, filename));
            });
        }
    });

    describe('gcc', () => {
        for (const filename of files.filter(x => x.startsWith('cfg-gcc'))) {
            it(filename, async () => {
                await DoCfgTest('g++', path.join(testcasespath, filename));
            });
        }
    });

    describe('clang', () => {
        for (const filename of files.filter(x => x.startsWith('cfg-clang'))) {
            it(filename, async () => {
                await DoCfgTest('clang', path.join(testcasespath, filename));
            });
        }
    });

    describe('msvc', () => {
        const msvcCompilerInfo = makeFakeCompilerInfo({
            group: 'vc',
            version: 'vc2022',
            compilerType: 'vc',
        });
        for (const filename of files.filter(x => x.includes('msvc'))) {
            it(filename, async () => {
                await DoCfgTest('vc', path.join(testcasespath, filename), false, msvcCompilerInfo);
            });
        }
    });

    describe('llvmir', () => {
        for (const filename of files.filter(x => x.includes('llvmir'))) {
            it(filename, async () => {
                await DoCfgTest('clang', path.join(testcasespath, filename), true);
            });
        }
    });

    describe('python', () => {
        const pythonCompilerInfo = makeFakeCompilerInfo({
            instructionSet: 'python',
            group: 'python3',
            version: 'Python 3.12.1',
            compilerType: 'python',
        });

        for (const filename of files.filter(x => x.includes('python'))) {
            it(filename, async () => {
                await DoCfgTest('python', path.join(testcasespath, filename), false, pythonCompilerInfo);
            });
        }
    });

    describe('xtensa', () => {
        // instructionSet is a real value, group/version/compilerType just need to be distinct from others
        const xtensaCompilerInfo = makeFakeCompilerInfo({
            instructionSet: 'xtensa',
            group: 'xtensa',
            version: 'xtensa',
            compilerType: 'xtensa',
        });

        for (const filename of files.filter(x => x.includes('xtensa'))) {
            it(filename, async () => {
                await DoCfgTest('python', path.join(testcasespath, filename), false, xtensaCompilerInfo);
            });
        }
    });

    describe('riscv', () => {
        for (const filename of files.filter(x => x.startsWith('cfg-riscv'))) {
            // As configured on godbolt.org: clang is detected from its version, while the rv32gcc/rv64gcc
            // groups have no dedicated cfg parser and use the base one
            const isClang = filename.includes('clang');
            const riscvCompilerInfo = makeFakeCompilerInfo({
                instructionSet: 'riscv64',
                group: isClang ? 'rv64clang' : 'rv64gcc',
                version: isClang ? 'clang' : 'gcc',
                compilerType: '',
            });
            it(filename, async () => {
                await DoCfgTest('', path.join(testcasespath, filename), false, riscvCompilerInfo);
            });
        }

        it.each([
            ['bgeu    a0, a1, .LBB0_2', InstructionType.conditionalJmpInst],
            ['bltz    a0, .L3', InstructionType.conditionalJmpInst],
            ['c.bnez  a0, .L3', InstructionType.conditionalJmpInst],
            ['beq     a0, zero, .LBB0_2', InstructionType.conditionalJmpInst],
            ['j       .L3', InstructionType.jmp],
            ['c.j     .LBB0_3', InstructionType.jmp],
            ['jal     zero, .LBB0_3', InstructionType.jmp],
            ['jal     x0, .LBB0_3', InstructionType.jmp],
            ['jr      a5', InstructionType.jmp],
            ['jalr    zero, 0(a5)', InstructionType.jmp],
            ['jalr    zero, a5, 0', InstructionType.jmp],
            ['jr      ra', InstructionType.retInst],
            ['jr      x1', InstructionType.retInst],
            ['c.jr    ra', InstructionType.retInst],
            ['jalr    zero, 0(ra)', InstructionType.retInst],
            ['jalr    x0, 0(x1)', InstructionType.retInst],
            ['jalr    zero, ra, 0', InstructionType.retInst],
            ['ret', InstructionType.retInst],
            ['mret', InstructionType.retInst],
            ['tail    foo', InstructionType.retInst],
            ['call    foo', InstructionType.notRetInst],
            ['jal     foo', InstructionType.notRetInst],
            ['jal     ra, foo', InstructionType.notRetInst],
            ['jalr    a5', InstructionType.notRetInst],
            ['c.jalr  a2', InstructionType.notRetInst],
            ['jalr    ra, 0(a2)', InstructionType.notRetInst],
            ['jalr    x1, 0(x12)', InstructionType.notRetInst],
            ['sb      a0, 0(sp)', InstructionType.notRetInst],
        ])('classifies %s', (inst, expected) => {
            expect(new RiscvInstructionSetInfo().getInstructionType(`        ${inst}`)).toEqual(expected);
        });
    });
});
