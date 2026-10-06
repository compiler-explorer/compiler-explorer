// Copyright (c) 2026, Compiler Explorer Authors
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

import {describe, expect, it} from 'vitest';

import {AsmRaw} from '../lib/parsers/asm-raw.js';

describe('AsmRaw', () => {
    // llvm-mc -filetype=obj, then objdump -d -l --insn-width=16 -r -C -M intel
    const objdump = [
        '',
        '/tmp/compiler-explorer-compilerXXXXXX/example.o:     file format elf64-x86-64',
        '',
        '',
        'Disassembly of section .text:',
        '',
        '0000000000000000 <test>:',
        'test():',
        '   0:\t89 0c 25 01 00 00 00                            \tmov    DWORD PTR ds:0x1,ecx',
        '   7:\te8 00 00 00 00                                  \tcall   c <test+0xc>',
        '\t\t\t8: R_X86_64_PLT32\tfoo-0x4',
        '   c:\tc3                                              \tret',
    ].join('\n');

    it('keeps addresses and opcodes', () => {
        const result = new AsmRaw().process(objdump, {});
        expect(result.asm[0]).toEqual({text: 'test:', source: null});
        expect(result.asm[1]).toMatchObject({address: 0, opcodes: ['89', '0c', '25', '01', '00', '00', '00']});
        expect(result.asm[4]).toMatchObject({address: 12, opcodes: ['c3']});
    });

    it('keeps relocations', () => {
        const result = new AsmRaw().process(objdump, {});
        expect(result.asm[3]).toEqual({text: '   R_X86_64_PLT32 foo-0x4', source: null});
    });

    it('passes a lone error message through', () => {
        const result = new AsmRaw().process('<No output: objdump returned 1>', {});
        expect(result.asm).toEqual([{text: '<No output: objdump returned 1>', source: null}]);
    });
});
