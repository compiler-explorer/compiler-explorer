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

import {AsmParser} from '../lib/parsers/asm-parser.js';
import {AsmParserM68k} from '../lib/parsers/asm-parser-m68k.js';

// Real m68k-gcc -O2 output containing inline asm with | source location markers.
// All m68k GCC variants (amigaos, human68k, elf, linux-gnu) produce this pattern.
const M68K_INLINE_ASM = [
    '#NO_APP',
    '\t.file\t"<stdin>"',
    '\t.text',
    '\t.align\t2',
    '\t.globl\ttest',
    '\t.type\ttest, @function',
    'test:',
    '\tmoveq #88,%d1',
    '#APP',
    '| 2 "<stdin>" 1',
    '\tmoveq #0x20,%d0',
    '\ttrap #15',
    '| 0 "" 2',
    '#NO_APP',
    '\trts',
    '\t.size\ttest, .-test',
    '\t.ident\t"GCC: (GNU) 13.4.0"',
].join('\n');

// Simple m68k output without inline asm
const M68K_SIMPLE = [
    '\t.file\t"example.c"',
    '\t.text',
    '\t.align\t2',
    '\t.globl\tadd',
    '\t.type\tadd, @function',
    'add:',
    '\tmove.l 4(%sp),%d0',
    '\tadd.l 8(%sp),%d0',
    '\trts',
    '\t.size\tadd, .-add',
].join('\n');

// Helper: check that any asm line contains the given substring
function hasLineContaining(result: {asm: {text: string}[]}, substring: string): boolean {
    return result.asm.some(line => line.text?.includes(substring));
}

// Helper: check that no asm line starts with the given prefix
function noLineStartsWith(result: {asm: {text: string}[]}, prefix: string): boolean {
    return !result.asm.some(line => line.text?.startsWith(prefix));
}

describe('AsmParserM68k tests', () => {
    const m68kParser = new AsmParserM68k();
    const baseParser = new AsmParser();

    describe('opcode identification', () => {
        it('should identify m68k opcodes', () => {
            expect(m68kParser.hasOpcode('\tmoveq #88,%d1')).toBe(true);
            expect(m68kParser.hasOpcode('\ttrap #15')).toBe(true);
            expect(m68kParser.hasOpcode('\tmove.l 4(%sp),%d0')).toBe(true);
            expect(m68kParser.hasOpcode('\trts')).toBe(true);
            expect(m68kParser.hasOpcode('\tjsr _LVORawDoFmt(%a6)')).toBe(true);
        });

        it('should not identify | comment lines as opcodes', () => {
            expect(m68kParser.hasOpcode('| 2 "<stdin>" 1')).toBe(false);
            expect(m68kParser.hasOpcode('| 0 "" 2')).toBe(false);
            expect(m68kParser.hasOpcode('| this is a comment')).toBe(false);
        });

        it('should not identify directives as opcodes', () => {
            expect(m68kParser.hasOpcode('\t.align\t2')).toBe(false);
            expect(m68kParser.hasOpcode('\t.globl\ttest')).toBe(false);
        });
    });

    describe('comment-only line filtering', () => {
        it('should treat | lines as comment-only', () => {
            const result = m68kParser.processAsm('| this is a m68k comment\n\trts', {commentOnly: true});
            expect(noLineStartsWith(result, '|')).toBe(true);
            expect(hasLineContaining(result, 'rts')).toBe(true);
        });

        it('should treat | source location markers as comment-only', () => {
            const result = m68kParser.processAsm('| 2 "<stdin>" 1\n\tmoveq #0x20,%d0', {commentOnly: true});
            expect(noLineStartsWith(result, '|')).toBe(true);
            expect(hasLineContaining(result, 'moveq #0x20,%d0')).toBe(true);
        });

        it('base parser should NOT filter | comment lines', () => {
            const result = baseParser.processAsm('| this is a m68k comment\n\trts', {commentOnly: true});
            expect(hasLineContaining(result, '| this is a m68k comment')).toBe(true);
        });
    });

    describe('inline asm source location markers', () => {
        it('should filter | source markers from inline asm with directives+comments enabled', () => {
            const result = m68kParser.processAsm(M68K_INLINE_ASM, {
                directives: true,
                commentOnly: true,
                labels: true,
            });

            // Source location markers should be gone
            expect(noLineStartsWith(result, '|')).toBe(true);

            // Actual opcodes should remain
            expect(hasLineContaining(result, 'moveq #88,%d1')).toBe(true);
            expect(hasLineContaining(result, 'moveq #0x20,%d0')).toBe(true);
            expect(hasLineContaining(result, 'trap #15')).toBe(true);
            expect(hasLineContaining(result, 'rts')).toBe(true);
        });

        it('base parser should leak | source markers through', () => {
            const result = baseParser.processAsm(M68K_INLINE_ASM, {
                directives: true,
                commentOnly: true,
                labels: true,
            });

            // Base parser does NOT recognise | as a comment, so markers leak through
            expect(noLineStartsWith(result, '|')).toBe(false);
        });
    });

    describe('simple m68k code', () => {
        it('should correctly parse simple m68k function with directives filtered', () => {
            const result = m68kParser.processAsm(M68K_SIMPLE, {
                directives: true,
                commentOnly: true,
                labels: true,
            });

            expect(hasLineContaining(result, 'add:')).toBe(true);
            expect(hasLineContaining(result, 'move.l 4(%sp),%d0')).toBe(true);
            expect(hasLineContaining(result, 'add.l 8(%sp),%d0')).toBe(true);
            expect(hasLineContaining(result, 'rts')).toBe(true);
        });

        it('should preserve all lines when no filters are applied', () => {
            const result = m68kParser.processAsm(M68K_SIMPLE, {
                directives: false,
                commentOnly: false,
                labels: false,
            });

            expect(hasLineContaining(result, '.file')).toBe(true);
            expect(hasLineContaining(result, '.globl')).toBe(true);
            expect(hasLineContaining(result, 'add:')).toBe(true);
            expect(hasLineContaining(result, 'rts')).toBe(true);
        });
    });
});
