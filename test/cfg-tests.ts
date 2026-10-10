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
import {
    InstructionType,
    LoongArchInstructionSetInfo,
    MipsInstructionSetInfo,
    RiscvInstructionSetInfo,
} from '../lib/cfg/instruction-sets/index.js';
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

    describe('mips', () => {
        for (const filename of files.filter(x => x.startsWith('cfg-mips'))) {
            const isClang = filename.includes('clang');
            const mipsCompilerInfo = makeFakeCompilerInfo({
                instructionSet: 'mips',
                group: isClang ? 'mips-clang' : filename.includes('gcc64') ? 'mips64' : 'mips',
                version: isClang ? 'clang' : 'gcc',
                compilerType: '',
            });
            it(filename, async () => {
                await DoCfgTest('', path.join(testcasespath, filename), false, mipsCompilerInfo);
            });
        }

        it.each([
            ['beq     $2,$0,$L6', InstructionType.conditionalJmpInst],
            ['bnez    $1, $BB0_2', InstructionType.conditionalJmpInst],
            ['blez    $5,.L4', InstructionType.conditionalJmpInst],
            ['bnel    $4,$5,$L3', InstructionType.conditionalJmpInst],
            ['bltc    $2,$4,$L6', InstructionType.conditionalJmpInst],
            ['bgeiuc  $a0,6,.L20', InstructionType.conditionalJmpInst],
            ['bc1t    $L5', InstructionType.conditionalJmpInst],
            ['bc1eqz  $f0,$L5', InstructionType.conditionalJmpInst],
            ['bnz.w   $w0,$L5', InstructionType.conditionalJmpInst],
            ['bbit0   $4,3,.L5', InstructionType.conditionalJmpInst],
            ['bteqz   $L6', InstructionType.conditionalJmpInst],
            ['beqz16  $4,$L6', InstructionType.conditionalJmpInst],
            ['b       $BB0_3', InstructionType.jmp],
            ['bc      .L3', InstructionType.jmp],
            ['j       $L3', InstructionType.jmp],
            ['jr      $25', InstructionType.jmp],
            ['jrc     $2', InstructionType.jmp],
            ['brsc    $a3', InstructionType.jmp],
            ['jalr    $0,$25', InstructionType.jmp],
            ['jr      $31', InstructionType.retInst],
            ['jr      $ra', InstructionType.retInst],
            ['j       $31', InstructionType.retInst],
            ['jrc     $ra', InstructionType.retInst],
            ['jr16    $ra', InstructionType.retInst],
            ['jalr    $zero, $ra', InstructionType.retInst],
            ['jraddiusp 32', InstructionType.retInst],
            ['restore.jrc 16,$ra,$gp', InstructionType.retInst],
            ['jal     ext(int)', InstructionType.notRetInst],
            ['jalr    $25', InstructionType.notRetInst],
            ['1:      jalr        $25', InstructionType.notRetInst],
            ['jalr    $31,$25', InstructionType.notRetInst],
            ['jalrc   $a3', InstructionType.notRetInst],
            ['bal     $L5', InstructionType.notRetInst],
            ['balc    _Z4ext2i', InstructionType.notRetInst],
            ['bgezal  $4,$L5', InstructionType.notRetInst],
            ['beqzalc $4,$L5', InstructionType.notRetInst],
            ['bnegi.w $w0,$w1,3', InstructionType.notRetInst],
            ['addiu   $sp,$sp,-32', InstructionType.notRetInst],
        ])('classifies %s', (inst, expected) => {
            expect(new MipsInstructionSetInfo().getInstructionType(`        ${inst}`)).toEqual(expected);
        });

        it.each([
            ['beq     $2,$0,$L6', true],
            ['bnel    $4,$5,$L3', true],
            ['b       $BB0_3', true],
            ['jr      $31', true],
            ['jal     ext(int)', true],
            ['jalr    $25', true],
            ['bc1eqz  $f0,$L5', true],
            ['bltc    $2,$4,$L6', false],
            ['beqzc   $4,$L13', false],
            ['bc      .L3', false],
            ['balc    _Z4ext2i', false],
            ['jrc     $31', false],
            ['jalrc   $a3', false],
            ['jraddiusp 32', false],
            ['bteqz   $L6', false],
            ['addiu   $sp,$sp,-32', false],
        ])('knows whether %s has a delay slot', (inst, expected) => {
            expect(new MipsInstructionSetInfo().hasDelaySlot(`        ${inst}`)).toEqual(expected);
        });
    });

    describe('loongarch', () => {
        for (const filename of files.filter(x => x.startsWith('cfg-loongarch'))) {
            const isClang = filename.includes('clang');
            const loongarchCompilerInfo = makeFakeCompilerInfo({
                instructionSet: 'loongarch',
                group: isClang ? 'loongarch64clang' : 'gccloongarch64',
                version: isClang ? 'clang' : 'gcc',
                compilerType: '',
            });
            it(filename, async () => {
                await DoCfgTest('', path.join(testcasespath, filename), false, loongarchCompilerInfo);
            });
        }

        it.each([
            ['bgt     $r4,$r12,.L6', InstructionType.conditionalJmpInst],
            ['blt     $a0, $a1, .LBB0_2', InstructionType.conditionalJmpInst],
            ['bgeu    $a1, $a2, .LBB0_4', InstructionType.conditionalJmpInst],
            ['bnez    $r12,.L9', InstructionType.conditionalJmpInst],
            ['blez    $a1, .LBB0_3', InstructionType.conditionalJmpInst],
            ['bceqz   $fcc0, .L3', InstructionType.conditionalJmpInst],
            ['b       .L3', InstructionType.jmp],
            ['jr      $r12', InstructionType.jmp],
            ['jr      $t8', InstructionType.jmp],
            ['jirl    $zero, $a0, 0', InstructionType.jmp],
            ['jr      $r1', InstructionType.retInst],
            ['jr      $ra', InstructionType.retInst],
            ['jirl    $zero, $ra, 0', InstructionType.retInst],
            ['jirl    $r0, $r1, 0', InstructionType.retInst],
            ['ret', InstructionType.retInst],
            ['tail36  $t8, foo', InstructionType.retInst],
            ['bl      %plt(_Z4ext2i)', InstructionType.notRetInst],
            ['jirl    $ra, $ra, 0', InstructionType.notRetInst],
            ['call36  foo', InstructionType.notRetInst],
            ['bstrpick.d $r4,$r4,31,0', InstructionType.notRetInst],
            ['bytepick.d $a0, $a1, $a2, 3', InstructionType.notRetInst],
            ['bitrev.w $a0, $a1', InstructionType.notRetInst],
        ])('classifies %s', (inst, expected) => {
            expect(new LoongArchInstructionSetInfo().getInstructionType(`        ${inst}`)).toEqual(expected);
        });
    });
});
