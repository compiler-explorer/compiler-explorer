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

import {describe, expect, it} from 'vitest';
import {AsmParser} from '../lib/parsers/asm-parser.js';
import {AsmEWAVRParser} from '../lib/parsers/asm-parser-ewavr.js';
import {SPIRVAsmParser} from '../lib/parsers/asm-parser-spirv.js';
import {VcAsmParser} from '../lib/parsers/asm-parser-vc.js';
import * as properties from '../lib/properties.js';

// Helper functions to reduce test duplication
function initializeParserAndFindLabels<T extends AsmParser>(
    ParserClass: new (...args: any[]) => T,
    constructorArgs: any[],
    asmLines: string[],
): Set<string> {
    const parser = new ParserClass(...constructorArgs);
    return parser.findUsedLabels(asmLines, true);
}

function processAsmWithParser<T extends AsmParser>(
    ParserClass: new (...args: any[]) => T,
    constructorArgs: any[],
    code: string,
    options: any,
) {
    const parser = new ParserClass(...constructorArgs);
    return parser.processAsm(code, options);
}

describe('AsmParser subclass compatibility', () => {
    describe('findUsedLabels method behavior', () => {
        it('should show EWAVR label finding now works correctly after refactoring', () => {
            const asmLines = [
                '_data:     .word 0x1234',
                '_main:',
                '           ldi r16, HIGH(_data)',
                '           ldi r17, LOW(_data)',
                '           call _subroutine',
                '           rjmp _main',
            ];
            const usedLabels = initializeParserAndFindLabels(AsmEWAVRParser, [properties.fakeProps({})], asmLines);

            // Fixed: now correctly finds labels in usage contexts after refactoring
            expect(usedLabels.has('_data')).toBe(true);
            expect(usedLabels.has('_subroutine')).toBe(true);
            expect(usedLabels.has('_main')).toBe(true);
            expect(usedLabels.has('HIGH')).toBe(true);
            expect(usedLabels.has('LOW')).toBe(true);
        });

        it('should show base class finds all identifier-like tokens as potential labels', () => {
            const asmLines = ['_start:', '    call printf', '    mov eax, value', '    jmp _start'];
            const usedLabels = initializeParserAndFindLabels(AsmParser, [], asmLines);

            // Base class finds ALL identifier-like tokens, not just actual labels
            expect(usedLabels.has('call')).toBe(true); // Instruction (not a label)
            expect(usedLabels.has('printf')).toBe(true); // Actual label reference
            expect(usedLabels.has('mov')).toBe(true); // Instruction (not a label)
            expect(usedLabels.has('eax')).toBe(true); // Register (not a label)
            expect(usedLabels.has('value')).toBe(true); // Actual label reference
            expect(usedLabels.has('jmp')).toBe(true); // Instruction (not a label)
            expect(usedLabels.has('_start')).toBe(true); // Actual label reference
        });
    });

    describe('processAsm custom implementations', () => {
        it('should verify VcAsmParser has different output format than base class', () => {
            const testCode = ['mov eax, OFFSET _data', 'call _function'].join('\n');
            const options = {directives: true, labels: false};

            const baseResult = processAsmWithParser(AsmParser, [], testCode, options);
            const vcResult = processAsmWithParser(VcAsmParser, [], testCode, options);

            // Base class returns labelDefinitions, VC parser does not
            expect(baseResult).toHaveProperty('labelDefinitions');
            expect(vcResult.labelDefinitions).toBeUndefined();

            // Base parser tracks label definitions it finds
            expect(baseResult.labelDefinitions).toEqual({});

            // Both should preserve the actual assembly instructions
            const baseMovLine = baseResult.asm.find(line => line.text?.includes('mov'));
            const vcMovLine = vcResult.asm.find(line => line.text?.includes('mov'));

            expect(baseMovLine?.text).toContain('OFFSET _data');
            expect(vcMovLine?.text).toContain('OFFSET _data');

            // Verify parsingTime and filteredCount are returned
            expect(typeof baseResult.parsingTime).toBe('number');
            expect(typeof baseResult.filteredCount).toBe('number');
        });

        it('should verify SPIRVAsmParser getUsedLabelsInLine detects percent labels', () => {
            const spirvCode = ['OpBranch %exit_label', '%exit_label = OpLabel', 'OpReturn'].join('\n');

            const result = processAsmWithParser(SPIRVAsmParser, [], spirvCode, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Should correctly identify SPIR-V %label syntax in definitions
            expect(result.labelDefinitions).toBeDefined();
            expect(result.labelDefinitions).toHaveProperty('%exit_label');
            expect(result.labelDefinitions!['%exit_label']).toBe(2); // Line number of definition

            // Should find the branch target in the labels array
            const branchLine = result.asm.find(line => line.text === 'OpBranch %exit_label');
            expect(branchLine).toBeDefined();
            expect(branchLine?.labels).toHaveLength(1);
            expect(branchLine?.labels?.[0].name).toBe('%exit_label');
            expect(branchLine?.labels?.[0].range).toBeDefined();
            expect(branchLine?.labels?.[0].range.startCol).toBeGreaterThan(0);
            const startCol = branchLine?.labels?.[0].range.startCol;
            expect(startCol).toBeDefined();
            expect(branchLine?.labels?.[0].range.endCol).toBeGreaterThan(startCol!);

            // OpReturn should have no labels
            const returnLine = result.asm.find(line => line.text === 'OpReturn');
            expect(returnLine?.labels).toHaveLength(0);
        });
    });
});
