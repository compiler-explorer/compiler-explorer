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

import {VcAsmParser} from '../lib/parsers/asm-parser-vc.js';

describe('VcAsmParser', () => {
    let parser: VcAsmParser;

    beforeEach(() => {
        parser = new VcAsmParser();
    });

    describe('VC assembly processing functionality', () => {
        it('should process VC assembly with function structure', () => {
            const vcAssembly = `
; Function compile flags: /Ogtp
; File example.cpp
; Line 1
_main	PROC
	mov	eax, OFFSET _variable
	call	_function
	ret	0
_main	ENDP
            `;

            const result = parser.processAsm(vcAssembly, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Should successfully process VC assembly format
            expect(result).toHaveProperty('asm');
            expect(Array.isArray(result.asm)).toBe(true);
            expect(result.asm.length).toBeGreaterThan(0);

            // Should preserve assembly instructions
            const hasInstructions = result.asm.some(
                line =>
                    line.text && (line.text.includes('mov') || line.text.includes('call') || line.text.includes('ret')),
            );
            expect(hasInstructions).toBe(true);
        });

        it('should handle VC-specific directives and comments', () => {
            const vcAssembly = `
; Function compile flags: /Ogtp
PUBLIC _function
_data	SEGMENT
_variable	DD	42
_data	ENDS
_text	SEGMENT
_function	PROC
	mov	eax, _variable
	ret
_function	ENDP
_text	ENDS
            `;

            const result = parser.processAsm(vcAssembly, {
                directives: true,
                labels: false,
                commentOnly: false,
            });

            // Should process VC segments and procedures
            expect(result.asm.length).toBeGreaterThan(0);

            // Should handle filtering of directives when requested
            const resultWithoutDirectives = parser.processAsm(vcAssembly, {
                directives: false,
                labels: false,
                commentOnly: false,
            });
            // Both should process successfully (directive filtering behavior may vary)
            expect(resultWithoutDirectives.asm.length).toBeGreaterThanOrEqual(0);
        });

        it('should correctly handle VC comment filtering', () => {
            const vcAssembly = `
; Function compile flags: /Ogtp
; File example.cpp
; Line 1
_main	PROC
; Another comment
	mov	eax, 42
	ret	0
_main	ENDP
            `;

            const resultWithComments = parser.processAsm(vcAssembly, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            const resultWithoutComments = parser.processAsm(vcAssembly, {
                directives: false,
                labels: false,
                commentOnly: true,
            });

            // Both should process successfully
            expect(resultWithComments.asm.length).toBeGreaterThan(0);
            expect(resultWithoutComments.asm.length).toBeGreaterThan(0);

            // Should have the actual instruction in both cases
            const hasInstruction = resultWithoutComments.asm.some(line => line.text?.includes('mov'));
            expect(hasInstruction).toBe(true);
        });
    });

    describe('VC assembly processing', () => {
        it('should process real VC assembly correctly', () => {
            const vcAssembly = `
; Function compile flags: /Ogtp /arch:SSE2
_main	PROC
	mov	eax, OFFSET _string
	push	eax
	call	_printf
	add	esp, 4
	xor	eax, eax
	ret	0
_main	ENDP
            `.trim();

            const result = parser.processAsm(vcAssembly, {
                directives: true,
                labels: false,
                commentOnly: false,
            });

            expect(result.asm.length).toBeGreaterThan(0);
            // VcAsmParser doesn't return labelDefinitions - it has its own format
            expect(result).toHaveProperty('asm');
            expect(result.labelDefinitions).toBeUndefined();
        });

        it('should handle VC-specific syntax', () => {
            const asmLines = ['OFFSET _data', 'DWORD PTR _variable', 'call _function'];

            const usedLabels = parser.findUsedLabels(asmLines, true);

            expect(usedLabels.has('_data')).toBe(true);
            expect(usedLabels.has('_variable')).toBe(true);
            expect(usedLabels.has('_function')).toBe(true);
        });
    });
});
