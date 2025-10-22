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

import {beforeEach, describe, expect, it} from 'vitest';

import {AsmEWAVRParser} from '../lib/parsers/asm-parser-ewavr.js';
import * as properties from '../lib/properties.js';

describe('AsmEWAVRParser', () => {
    let parser: AsmEWAVRParser;

    beforeEach(() => {
        parser = new AsmEWAVRParser(properties.fakeProps({}));
    });

    describe('EWAVR assembly processing functionality', () => {
        it('should process EWAVR assembly and preserve AVR instruction formats', () => {
            const ewavrAssembly = [
                '_main:',
                '                ldi r16, 0xFF',
                '                call _function',
                '                ret',
            ].join('\n');

            const result = parser.processAsm(ewavrAssembly, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Should preserve specific EWAVR/AVR instruction formats
            const ldiInstruction = result.asm.find(line => line.text?.includes('ldi'));
            expect(ldiInstruction?.text).toContain('r16, 0xFF');

            const callInstruction = result.asm.find(line => line.text?.includes('call'));
            expect(callInstruction?.text).toContain('_function');

            // AsmEWAVRParser has custom processAsm that doesn't return labelDefinitions
            expect(result.labelDefinitions).toBeUndefined();
        });

        it('should handle EWAVR-specific directives filtering', () => {
            const ewavrAssembly = [
                'RSEG CODE',
                'PUBLIC _function',
                '_function:',
                '                ldi r16, HIGH(_external_var)',
                '                END',
            ].join('\n');

            const resultWithDirectives = parser.processAsm(ewavrAssembly, {
                directives: false, // Include directives
                labels: false,
                commentOnly: false,
            });

            const resultFilteringDirectives = parser.processAsm(ewavrAssembly, {
                directives: true, // Filter out directives
                labels: false,
                commentOnly: false,
            });

            // When filtering directives, should have fewer lines
            expect(resultFilteringDirectives.asm.length).toBeLessThan(resultWithDirectives.asm.length);

            // Should preserve the HIGH() function call format
            const hasHighFunction = resultWithDirectives.asm.some(line => line.text?.includes('HIGH('));
            expect(hasHighFunction).toBe(true);
        });

        it('should filter EWAVR comments when commentOnly is true', () => {
            const ewavrAssembly = [
                '// This is a comment',
                '_main:',
                '                ldi r16, 42',
                '                ret',
            ].join('\n');

            const resultWithComments = parser.processAsm(ewavrAssembly, {
                directives: false,
                labels: false,
                commentOnly: false, // Include comments
            });

            const resultFilteringComments = parser.processAsm(ewavrAssembly, {
                directives: false,
                labels: false,
                commentOnly: true, // Filter out comments
            });

            // When filtering comments, should have fewer lines
            expect(resultFilteringComments.asm.length).toBeLessThan(resultWithComments.asm.length);

            // Should still have the actual instruction
            const hasLdiInstruction = resultFilteringComments.asm.some(line => line.text?.includes('ldi'));
            expect(hasLdiInstruction).toBe(true);

            // Should not have comment lines when filtering
            const hasCommentLine = resultFilteringComments.asm.some(line => line.text?.startsWith('//'));
            expect(hasCommentLine).toBe(false);
        });
    });

    describe('EWAVR assembly processing', () => {
        it('should preserve EWAVR AVR instruction operands and labels', () => {
            const ewavrAssembly = [
                '_main:',
                '                ldi r16, 0xFF',
                '                out PORTB, r16',
                '                call _delay',
                'delay_loop:',
                '                brne delay_loop',
                '                ret',
            ].join('\n');

            const result = parser.processAsm(ewavrAssembly, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Should preserve specific AVR instruction formats
            const outInstruction = result.asm.find(line => line.text?.includes('out'));
            expect(outInstruction?.text).toContain('PORTB, r16');

            const branchInstruction = result.asm.find(line => line.text?.includes('brne'));
            expect(branchInstruction?.text).toContain('delay_loop');

            // AsmEWAVRParser has custom processAsm format
            expect(result.labelDefinitions).toBeUndefined();
        });

        it('should correctly find labels in usage contexts after refactoring fix', () => {
            const asmLines = ['ldi r16, HIGH(_data)', 'ldi r17, LOW(_data)', 'call _subroutine', 'rjmp _loop'];

            const usedLabels = parser.findUsedLabels(asmLines, true);

            // After refactoring: correctly finds labels in usage contexts
            expect(usedLabels.has('_data')).toBe(true);
            expect(usedLabels.has('_subroutine')).toBe(true);
            expect(usedLabels.has('_loop')).toBe(true);
            expect(usedLabels.has('HIGH')).toBe(true); // Ensure HIGH is included
            expect(usedLabels.has('LOW')).toBe(true); // Ensure LOW is included
            // Verify we found the expected labels rather than checking exact count
        });

        it('should handle EWAVR segment syntax and register operations', () => {
            const ewavrCode = [
                "CODE32 segment 'CODE'",
                'public _main',
                '_main:',
                '                    ldi r16, 0xFF',
                '                    out DDRC, r16',
                '                CODE32 ends',
            ].join('\n');

            const result = parser.processAsm(ewavrCode, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Should preserve AVR register operations
            const ldiInstruction = result.asm.find(line => line.text?.includes('ldi'));
            expect(ldiInstruction?.text).toContain('r16, 0xFF');

            const outInstruction = result.asm.find(line => line.text?.includes('out'));
            expect(outInstruction?.text).toContain('DDRC, r16');

            // AsmEWAVRParser uses custom format
            expect(result.labelDefinitions).toBeUndefined();
        });
    });
});
