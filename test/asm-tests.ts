// Copyright (c) 2018, Compiler Explorer Authors
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

import {unwrap} from '../lib/assert.js';
import {VcAsmParser} from '../lib/parsers/asm-parser-vc.js';
import {AsmParserZ88dk} from '../lib/parsers/asm-parser-z88dk.js';
import {AsmParser} from '../lib/parsers/asm-parser.js';
import {AsmRegex} from '../lib/parsers/asmregex.js';

import {makeFakeParseFiltersAndOutputOptions} from './utils.js';

describe('ASM CL parser', () => {
    it('should work for error documents', () => {
        const parser = new VcAsmParser();
        const result = parser.process('<Compilation failed>', {
            directives: true,
            labels: false,
            libraryCode: false,
            commentOnly: false,
            trim: false,
            optOutput: false,
            binary: false,
            binaryObject: false,
            execute: false,
            demangle: false,
            intel: false,
            debugCalls: false,
        });

        expect(result.asm).toEqual([
            {
                source: null,
                text: '<Compilation failed>',
            },
        ]);
    });
});

describe('ASM regex base class', () => {
    it('should leave unfiltered lines alone', () => {
        const line = '     this    is    a line';
        expect(AsmRegex.filterAsmLine(line, makeFakeParseFiltersAndOutputOptions({}))).toEqual(line);
    });
    it('should use up internal whitespace when asked', () => {
        expect(
            AsmRegex.filterAsmLine('     this    is    a line', makeFakeParseFiltersAndOutputOptions({trim: true})),
        ).toEqual('  this is a line');
        expect(
            AsmRegex.filterAsmLine('this    is    a line', makeFakeParseFiltersAndOutputOptions({trim: true})),
        ).toEqual('this is a line');
    });
    it('should keep whitespace in strings', () => {
        expect(
            AsmRegex.filterAsmLine('equs     "this    string"', makeFakeParseFiltersAndOutputOptions({trim: true})),
        ).toEqual('equs "this    string"');
        expect(
            AsmRegex.filterAsmLine(
                '     equs     "this    string"',
                makeFakeParseFiltersAndOutputOptions({trim: true}),
            ),
        ).toEqual('  equs "this    string"');
        expect(
            AsmRegex.filterAsmLine(
                'equs     "this    \\"  string  \\""',
                makeFakeParseFiltersAndOutputOptions({trim: true}),
            ),
        ).toEqual('equs "this    \\"  string  \\""');
    });
    it('should not get upset by mismatched strings', () => {
        expect(
            AsmRegex.filterAsmLine('a   "string    \'yeah', makeFakeParseFiltersAndOutputOptions({trim: true})),
        ).toEqual('a "string \'yeah');
    });
});

describe('ASM parser base class', () => {
    let parser;
    const filters = {};

    beforeAll(() => {
        parser = new AsmParser();
    });

    it('should recognize source column numbers', () => {
        const asm = `
    .text
    .intel_syntax noprefix
    .file	"tmp.cpp"
    .file	1 "/usr/include" "stdlib.h"
    .file	2 "/usr/bin/../lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/bits" "std_abs.h"
    .file	3 "/usr/bin/../lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9" "cstdlib"
    .file	4 "/usr/lib/llvm-11/lib/clang/11.0.0/include" "stddef.h"
    .file	5 "/usr/bin/../lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9" "stdlib.h"
    .globl	main                            # -- Begin function main
    .p2align	4, 0x90
    .type	main,@function
main:                                   # @main
    .Lfunc_begin0:
    .file	6 "/home/necto/proj/compiler-explorer" "tmp.cpp"
    .loc	6 3 0                           # tmp.cpp:3:0
    .cfi_startproc
# %bb.0:                                # %entry
    push	rbp
    .cfi_def_cfa_offset 16
    .cfi_offset rbp, -16
    mov1	rbp, rsp
    .cfi_def_cfa_register rbp
    sub	rsp, 48
    mov2	dword ptr [rbp - 4], 0
.Ltmp0:
    .loc	6 4 20 prologue_end             # tmp.cpp:4:20
    mov3	edi, 16
    call	malloc
    .loc	6 4 9 is_stmt 0                 # tmp.cpp:4:9
    mov4	qword ptr [rbp - 16], rax
`;
        const output = parser.process(asm, filters);
        const push_line = output.asm.find(line => line.text.trim().startsWith('push'));
        const mov1_line = output.asm.find(line => line.text.trim().startsWith('mov1'));
        const call_line = output.asm.find(line => line.text.trim().startsWith('call'));
        const mov4_line = output.asm.find(line => line.text.trim().startsWith('mov4'));
        expect(push_line.source).not.toHaveProperty('column');
        expect(mov1_line.source).not.toHaveProperty('column');
        expect(call_line.source.column).toEqual(20);
        expect(mov4_line.source.column).toEqual(9);
    });

    it('should parse line numbers when a column is not specified', () => {
        const asm = `
        .section .text
.LNDBG_TX:
# mark_description "Intel(R) C Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 12.1 Build 20120410";
        .file "iccKTGaIssTdIn_"
        .text
..TXTST0:
# -- Begin  main
# mark_begin;
       .align    16,0x90
        .globl main
main:
..B1.1:                         # Preds ..B1.0
..___tag_value_main.2:                                          #
..LN0:
  .file   1 "-"
   .loc    1  2  is_stmt 1
        pushq     %rbp                                          #2.12
`;
        const output = parser.process(asm, filters);
        const pushq_line = output.asm.find(line => line.text.trim().startsWith('pushq'));
        expect(pushq_line.source).not.toHaveProperty('column');
        expect(pushq_line.source.line).toEqual(2);
    });
});

describe('ASM parser', () => {
    let parser: AsmParser;
    const filters = {};

    beforeAll(() => {
        parser = new AsmParser();
    });

    it('should not parse slowly', () => {
        const asm = `
._square
${' '.repeat(65530)}x
${' '.repeat(65530)}x
${' '.repeat(65530)}x
${' '.repeat(65530)}x
        pop     bc
        pop     hl
        push    hl
        push    bc
        ld      d,h
        ld      e,l
        call    l_mult
        ret
`;
        const output = parser.process(asm, filters);
        expect(parseInt(unwrap(output.parsingTime))).toBeLessThan(500); // reported as ms, generous timeout for ci runner
    });
});

describe('ASM parser z88dk', () => {
    let parser: AsmParserZ88dk;
    const filters = {};

    beforeAll(() => {
        parser = new AsmParserZ88dk(undefined as any);
    });

    it('should not parse slowly', () => {
        const asm = `
._square
${' '.repeat(65530)}x
${' '.repeat(65530)}x
${' '.repeat(65530)}x
${' '.repeat(65530)}x
        pop     bc
        pop     hl
        push    hl
        push    bc
        ld      d,h
        ld      e,l
        call    l_mult
        ret
`;
        const output = parser.process(asm, filters);
        expect(parseInt(unwrap(output.parsingTime))).toBeLessThan(500); // reported as ms, generous timeout for ci runner
    });
});
