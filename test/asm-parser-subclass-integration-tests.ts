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

import {AsmEWAVRParser} from '../lib/parsers/asm-parser-ewavr.js';
import {SPIRVAsmParser} from '../lib/parsers/asm-parser-spirv.js';
import {VcAsmParser} from '../lib/parsers/asm-parser-vc.js';
import {AsmParser} from '../lib/parsers/asm-parser.js';
import * as properties from '../lib/properties.js';

describe('AsmParser subclass compatibility', () => {
    describe('labelFindFor method behavior', () => {
        it('should use VcAsmParser labelFindFor override to find specific VC labels', () => {
            const parser = new VcAsmParser();
            const asmLines = ['_start:', 'mov eax, OFFSET _data', 'call _function', 'jmp _start'];

            const usedLabels = parser.findUsedLabels(asmLines, true);

            // VC-specific label detection should find these labels
            expect(usedLabels.has('_data')).toBe(true);
            expect(usedLabels.has('_function')).toBe(true);
            expect(usedLabels.has('_start')).toBe(true);
        });

        it('should demonstrate EWAVR labelFindFor returns definition regex instead of usage regex', () => {
            const parser = new AsmEWAVRParser(properties.fakeProps({}));
            const asmLines = ['main:', 'ldi r16, HIGH(data_table)', 'call subroutine', 'rjmp main'];

            // EWAVR's labelFindFor() returns labelDef (for definitions with :) instead of a usage finder
            const labelFindRegex = parser.labelFindFor();
            expect(labelFindRegex.source).toBe('^`?\\?*<?([A-Z_a-z][\\w :]*)>?`?:$');

            // This causes findUsedLabels to find no labels because it's looking for definitions not usage
            const usedLabels = parser.findUsedLabels(asmLines, true);
            expect(usedLabels.size).toBe(0);
        });

        it('should show base class labelFindFor returns usage regex', () => {
            const parser = new AsmParser();
            const asmLines = ['main:', 'mov r1, #value', 'bl function', 'b main'];

            // Base class returns a usage finder regex (takes asmLines parameter)
            const labelFindRegex = parser.labelFindFor(asmLines);
            expect(labelFindRegex.global).toBe(true); // Should be global for finding multiple matches
            expect(labelFindRegex.source).toBe('[.A-Z_a-z][\\w$.]*'); // Should match label usage

            // This correctly finds used labels
            const usedLabels = parser.findUsedLabels(asmLines, true);
            expect(usedLabels.has('value')).toBe(true);
            expect(usedLabels.has('function')).toBe(true);
            expect(usedLabels.has('main')).toBe(true);
        });
    });

    describe('processAsm custom implementations', () => {
        it('should verify VcAsmParser has different output format than base class', () => {
            const testCode = 'mov eax, OFFSET _data';

            const baseParser = new AsmParser();
            const vcParser = new VcAsmParser();

            const baseResult = baseParser.processAsm(testCode, {directives: true, labels: false});
            const vcResult = vcParser.processAsm(testCode, {directives: true, labels: false});

            // Base class returns labelDefinitions, VC parser does not
            expect(baseResult).toHaveProperty('labelDefinitions');
            expect(vcResult.labelDefinitions).toBeUndefined();

            // Both should process the assembly successfully
            expect(baseResult.asm.length).toBeGreaterThan(0);
            expect(vcResult.asm.length).toBeGreaterThan(0);
        });

        it('should verify SPIRVAsmParser getUsedLabelsInLine detects percent labels', () => {
            const spirvParser = new SPIRVAsmParser();

            const spirvCode = 'OpBranch %exit_label\n%exit_label = OpLabel';

            const result = spirvParser.processAsm(spirvCode, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Should correctly identify SPIR-V %label syntax
            expect(result.labelDefinitions).toHaveProperty('%exit_label');

            // Should find the branch target
            const hasBranchTarget = result.asm.some(line => line.labels?.some(label => label.name === '%exit_label'));
            expect(hasBranchTarget).toBe(true);
        });
    });
});
