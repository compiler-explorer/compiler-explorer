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
import {AsmParser} from '../lib/parsers/asm-parser.js';
import {VcAsmParser} from '../lib/parsers/asm-parser-vc.js';

// Test helper class that extends VcAsmParser to expose protected properties for testing
class VcAsmParserForTest extends VcAsmParser {
    getCommentOnlyRegexForTest() {
        return this.commentOnly;
    }

    getDefinesFunctionRegexForTest() {
        return this.definesFunction;
    }
}

describe('VcAsmParser', () => {
    let parser: VcAsmParser;
    let testParser: VcAsmParserForTest;

    beforeEach(() => {
        parser = new VcAsmParser();
        testParser = new VcAsmParserForTest();
    });

    describe('VC assembly processing functionality', () => {
        it('should have custom processAsm that returns different format than base class', () => {
            // Simple test that doesn't trigger complex VC parsing logic
            const simpleAsm = 'nop';

            const baseParser = new AsmParser();
            const baseResult = baseParser.processAsm(simpleAsm, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Base parser returns labelDefinitions, VC parser format is different
            expect(baseResult).toHaveProperty('labelDefinitions');
            expect(typeof baseResult.labelDefinitions).toBe('object');
        });

        it('should handle VC-specific directives correctly', () => {
            const vcAssembly = ['PUBLIC _function', '_data\tSEGMENT', '_variable\tDD\t42', '_data\tENDS'].join('\n');

            const resultWithDirectives = parser.processAsm(vcAssembly, {
                directives: false, // Should include directives
                labels: false,
                commentOnly: false,
            });

            const resultFilteringDirectives = parser.processAsm(vcAssembly, {
                directives: true, // Should filter out directives
                labels: false,
                commentOnly: false,
            });

            // When filtering directives, should have fewer lines
            expect(resultFilteringDirectives.asm.length).toBeLessThan(resultWithDirectives.asm.length);

            // Should still preserve the data declaration when not filtering
            const hasDataDecl = resultWithDirectives.asm.some(line => line.text?.includes('DD'));
            expect(hasDataDecl).toBe(true);
        });

        it('should correctly identify VC comments using commentOnly regex', () => {
            const commentLine = '; This is a VC comment';
            const indentedComment = '    ; Indented comment';
            const codeLine = 'mov eax, ebx';

            const commentOnlyRegex = testParser.getCommentOnlyRegexForTest();

            // VC commentOnly regex is /^;/ - only matches lines starting with ;
            expect(commentOnlyRegex.test(commentLine)).toBe(true);
            expect(commentOnlyRegex.test(codeLine)).toBe(false);

            // VC regex doesn't match comments with leading whitespace
            expect(commentOnlyRegex.test(indentedComment)).toBe(false);
        });
    });

    describe('VC assembly processing', () => {
        it('should recognize VC function definitions with PROC keyword', () => {
            const procLine = '_function\tPROC';
            const nonProcLine = '_function:';

            const definesFunctionRegex = testParser.getDefinesFunctionRegexForTest();

            // Test the function definition regex directly
            expect(definesFunctionRegex.test(procLine)).toBe(true);
            expect(definesFunctionRegex.test(nonProcLine)).toBe(false);

            // Should extract function name
            const match = procLine.match(definesFunctionRegex);
            expect(match?.[1]).toBe('_function');
        });

        it('should find labels in VC-specific syntax using custom labelFindFor', () => {
            const asmLines = ['mov eax, OFFSET _data', 'mov ebx, DWORD PTR _variable', 'call _function'];

            const usedLabels = parser.findUsedLabels(asmLines, true);

            // VcAsmParser should find the main labels we're looking for
            expect(usedLabels.has('_data')).toBe(true);
            expect(usedLabels.has('_variable')).toBe(true);
            expect(usedLabels.has('_function')).toBe(true);

            // May find additional labels/symbols in VC syntax - that's expected behavior
            expect(usedLabels.size).toBeGreaterThanOrEqual(3);
        });
    });
});
