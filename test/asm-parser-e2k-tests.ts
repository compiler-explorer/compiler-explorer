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

import {E2KAsmParser, MCSTLCCE2KAsmParser} from '../lib/parsers/asm-parser-e2k.js';
import type {ParsedAsmResult} from '../types/asmresult/asmresult.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';

describe('E2KAsmParser tests', () => {
    const e2kParser = new E2KAsmParser();

    it('should expand first tab to 2 spaces (LCC)', () => {
        const input = '\t{\n\t  nop\t1\n\t  return\t%ctpr3\n\t}';
        const result = e2kParser.processAsm(input, {});
        const lines = result.asm.map(line => line.text);
        expect(lines[0]).toBe('  {');
        expect(lines[1]).toBe('    nop 1');
        expect(lines[2]).toBe('    return      %ctpr3');
        expect(lines[3]).toBe('  }');
    });

    it('should expand first tab to 2 spaces (LLVM)', () => {
        const input = '{\n\tnop\t1\n\treturn\t%ctpr3\n}';
        const result = e2kParser.processAsm(input, {});
        const lines = result.asm.map(line => line.text);
        expect(lines[0]).toBe('{');
        expect(lines[1]).toBe('  nop   1');
        expect(lines[2]).toBe('  return        %ctpr3');
        expect(lines[3]).toBe('}');
    });

    it('should treat ! lines as comment-only', () => {
        const input = '! <0000>\n\tnop';
        const result = e2kParser.processAsm(input, {commentOnly: true});
        const lines = result.asm.map(line => line.text);
        expect(lines[0]).toBe('  nop');
    });

    function expectParsedAsmResult(result: ParsedAsmResult, expected: ParsedAsmResult): void {
        expect(result.labelDefinitions).toEqual(expected.labelDefinitions);
        expect(result.asm).toHaveLength(expected.asm.length);
        for (let i = 0; i < result.asm.length; i++) {
            expect(result.asm[i]).toMatchObject(expected.asm[i]);
        }
    }

    it('should identify bundle addresses, instructions and relocations', () => {
        const input = `

example.o:     file format elf64-e2k


Disassembly of section .text:

0000000000000000 <foo>:
foo():
/tmp/example.c:5
   0:
  getsp,0 _f32s 0xfffffff0, %dr4
  disp %ctpr1, 0x0
  setwd wsz = 0x8, nfx = 0x1, dbl = 0x0
  setbn rbs = 0x4, rsz = 0x3, rcur = 0x0

			8: R_E2K_DISP	bar+0x8
  20:
  nop 2
  ldd,2 0x0, _f64 0x8, %db[0]

			28: R_E2K_64_ABS_LIT	global
  30:
  addd,2,sm 0x0, %dr0, %db[1]

/tmp/example.c:6
  38:
  ipd 3
  muls,0 %r1, %r2, %r1
  call %ctpr1, wbs = 0x4

  50:
  nop 4
  return %ctpr3

  58:
  adds,3 %r1, %b[0], %r1

  60:
  ct %ctpr3
  ipd 3
  sxt,3 0x2, %r1, %dr0`;

        const s5 = {file: null, line: 5};
        const s6 = {file: null, line: 6};
        const expected: ParsedAsmResult = {
            asm: [
                {text: 'foo:'},
                {address: 0x00, source: s5, text: '  getsp,0 _f32s 0xfffffff0, %dr4'},
                {address: 0x00, source: s5, text: '  disp %ctpr1, 0x0'},
                {address: 0x00, source: s5, text: '  setwd wsz = 0x8, nfx = 0x1, dbl = 0x0'},
                {address: 0x00, source: s5, text: '  setbn rbs = 0x4, rsz = 0x3, rcur = 0x0'},
                {text: ''},
                {address: 0x08, text: '  R_E2K_DISP bar+0x8'},
                {address: 0x20, source: s5, text: '  nop 2'},
                {address: 0x20, source: s5, text: '  ldd,2 0x0, _f64 0x8, %db[0]'},
                {text: ''},
                {address: 0x28, text: '  R_E2K_64_ABS_LIT global'},
                {address: 0x30, source: s5, text: '  addd,2,sm 0x0, %dr0, %db[1]'},
                {text: ''},
                {address: 0x38, source: s6, text: '  ipd 3'},
                {address: 0x38, source: s6, text: '  muls,0 %r1, %r2, %r1'},
                {address: 0x38, source: s6, text: '  call %ctpr1, wbs = 0x4'},
                {text: ''},
                {address: 0x50, source: s6, text: '  nop 4'},
                {address: 0x50, source: s6, text: '  return %ctpr3'},
                {text: ''},
                {address: 0x58, source: s6, text: '  adds,3 %r1, %b[0], %r1'},
                {text: ''},
                {address: 0x60, source: s6, text: '  ct %ctpr3'},
                {address: 0x60, source: s6, text: '  ipd 3'},
                {address: 0x60, source: s6, text: '  sxt,3 0x2, %r1, %dr0'},
            ],
            labelDefinitions: {
                foo: 1,
            },
        };

        const filters: Partial<ParseFiltersAndOutputOptions> = {binaryObject: true};
        const result = e2kParser.processAsm(input, filters);
        expectParsedAsmResult(result, expected);
    });
});

describe('MCSTLCCE2KAsmParser tests', () => {
    const lccParser = new MCSTLCCE2KAsmParser();

    describe('source comment filtering', () => {
        it('should remove only source comment lines', () => {
            const input = '! /tmp/file.c : 1\n\tnop 7 ! /tmp/file.c : 2';
            const result = lccParser.processAsm(input, {commentOnly: true});
            const lines = result.asm.map(line => line.text);
            expect(lines[0]).toBe('  nop 7');
        });
        it('should strip source comments', () => {
            const input = 'label: ! /tmp/file.c : 1\n\tct %ctpr1 !  /tmp/file.c  :  2  ';
            const result = lccParser.processAsm(input, {commentOnly: true});
            const lines = result.asm.map(line => line.text);
            expect(lines[0]).toBe('label:');
            expect(lines[1]).toBe('  ct %ctpr1');
        });
    });

    describe('label defines', () => {
        it('without source comment', () => {
            const input = 'entry:\n\tnop';
            const result = lccParser.processAsm(input, {commentOnly: true});
            const lines = result.asm.map(line => line.text);
            expect(lines[0]).toBe('entry:');
            expect(lines[1]).toBe('  nop');
        });
        it('with source comment', () => {
            const input = 'entry:  ! /example.c : 1\n\tnop';
            const result = lccParser.processAsm(input, {commentOnly: true});
            const lines = result.asm.map(line => line.text);
            expect(lines[0]).toBe('entry:');
            expect(lines[1]).toBe('  nop');
        });
    });
});
