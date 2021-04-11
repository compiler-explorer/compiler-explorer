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

import { AsmParser } from '../lib/asm-parser';
import { VcAsmParser } from '../lib/asm-parser-vc';
import { AsmRegex } from '../lib/asmregex';

describe('ASM CL parser', () => {
    it('should work for error documents', () => {
        const parser = new VcAsmParser();
        const result = parser.process('<Compilation failed>', {
            directives: true,
        });

        result.asm.should.deep.equal([{
            source: null,
            text: '<Compilation failed>',
        }]);
    });
});

describe('ASM regex base class', () => {
    it('should leave unfiltered lines alone', () => {
        const line = '     this    is    a line';
        AsmRegex.filterAsmLine(line, {}).should.equal(line);
    });
    it('should use up internal whitespace when asked', () => {
        AsmRegex.filterAsmLine('     this    is    a line', {trim: true}).should.equal('  this is a line');
        AsmRegex.filterAsmLine('this    is    a line', {trim: true}).should.equal('this is a line');
    });
    it('should keep whitespace in strings', () => {
        AsmRegex.filterAsmLine('equs     "this    string"', {trim: true}).should.equal('equs "this    string"');
        AsmRegex.filterAsmLine('     equs     "this    string"', {trim: true}).should.equal('  equs "this    string"');
        AsmRegex.filterAsmLine('equs     "this    \\"  string  \\""', {trim: true}).should.equal('equs "this    \\"  string  \\""');
    });
    it('should not get upset by mismatched strings', () => {
        AsmRegex.filterAsmLine("a   \"string    'yeah", {trim: true}).should.equal("a \"string 'yeah");
    });
});

describe('ASM parser base class', () => {
    let parser;
    const filters = {};

    before(() => {
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
        push_line.source.should.not.have.ownProperty('column');
        mov1_line.source.should.not.have.ownProperty('column');
        call_line.source.column.should.equal(20);
        mov4_line.source.column.should.equal(9);
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
        pushq_line.source.should.not.have.ownProperty('column');
        pushq_line.source.line.should.equal(2);
    });
});
