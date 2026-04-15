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

import {type AssemblyInstructionInfo} from '../../types/assembly-docs.interfaces.js';
import {
    AssemblySyntax,
    ATT_SYNTAX_WARNING,
    addAttSyntaxWarningIfNeeded,
    determineAssemblySyntax,
} from '../assembly-syntax.js';

const mockUrl = 'https://example.com';
function makeInfo(tooltip: string, html?: string): AssemblyInstructionInfo {
    html = html ?? `<p>${tooltip}</p>`;
    return {tooltip, html, url: mockUrl};
}

describe(addAttSyntaxWarningIfNeeded, () => {
    describe('warns when syntax is att and', () => {
        const attSyntax: AssemblySyntax = 'att';
        it('appends warning to tooltip and not html when only tooltip references cardinality operand and source/destination', () => {
            const data = makeInfo('first operand (destination)', '<p>Simple description</p>');
            const result = addAttSyntaxWarningIfNeeded(data, attSyntax);

            expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
            expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
        });
        it('appends warning to html and not tooltip only html references cardinality operand and source/destination', () => {
            const data = makeInfo('Simple tooltip', '<p>second operand (source)</p>');
            const result = addAttSyntaxWarningIfNeeded(data, attSyntax);

            expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
            expect(result.html).toContain(ATT_SYNTAX_WARNING);
        });
        it('both tooltip and html reference cardinality and source/destination', () => {
            const text = 'copies the first operand (source) to the second operand (destination)';
            const data = makeInfo(text);
            const result = addAttSyntaxWarningIfNeeded(data, attSyntax);
            expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
            expect(result.html).toContain(ATT_SYNTAX_WARNING);
        });
        it('handles third and fourth operand references', () => {
            const haddpsDescription =
                'Adds single precision floating-point values in the third and fourth dword of the destination operand and stores the result in the second dword of the destination operand.';
            const data = makeInfo(haddpsDescription);
            const result = addAttSyntaxWarningIfNeeded(data, 'att');
            expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
            expect(result.html).toContain(ATT_SYNTAX_WARNING);
        });
    });
    describe('does not warn when', () => {
        describe('syntax is intel', () => {
            it('and references cardinality of operands', () => {
                const data = makeInfo('first operand (destination)');
                const result = addAttSyntaxWarningIfNeeded(data, 'intel');
                expect(result).toEqual(data);
            });
        });

        describe('syntax is att and', () => {
            const attSyntax = 'att';
            it('no cardinality references to destination/source operands', () => {
                const data = makeInfo(
                    'Decrements the stack pointer and then stores the source operand on the top of the stack',
                );
                const result = addAttSyntaxWarningIfNeeded(data, attSyntax);

                expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
                expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
            });
            it('cardinality is present but source/destination is absent', () => {
                const data = makeInfo('Performs a signed multiplication of two operands');
                const result = addAttSyntaxWarningIfNeeded(data, attSyntax);

                expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
                expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
            });
            it('source/destination is present but cardinality is absent', () => {
                const popDesc =
                    'Loads the value from the top of the stack to the location specified with the destination operand (or explicit opcode) and then increments the stack pointer.';
                const data = makeInfo(popDesc);
                const result = addAttSyntaxWarningIfNeeded(data, attSyntax);
                expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
                expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
            });
            it('tooltip contains "operand" and html contains "destination"', () => {
                const data = makeInfo('The first operand is foo', '<p>Copies bar to the destination register</p>');
                const result = addAttSyntaxWarningIfNeeded(data, 'att');
                expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
                expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
            });
            it('tooltip contains "source" and html contains "operand"', () => {
                const data = makeInfo(
                    'The source operand is foo',
                    '<p>Copies bar to the last destination register</p>',
                );
                const result = addAttSyntaxWarningIfNeeded(data, 'att');
                expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
                expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
            });
        });
    });

    describe('purity', () => {
        it('does not mutate the original data object regardless of syntax', () => {
            const data = makeInfo('first operand (destination)');
            const originalTooltip = data.tooltip;
            const originalHtml = data.html;

            addAttSyntaxWarningIfNeeded(data, 'att');
            expect(data.tooltip).toBe(originalTooltip);
            expect(data.html).toBe(originalHtml);
        });
        it('preserves the url field regardless of syntax', () => {
            let result: AssemblyInstructionInfo;
            const data = makeInfo('the first operand (destination)');

            result = addAttSyntaxWarningIfNeeded(data, 'att');
            expect(result.url).toBe(mockUrl);
            result = addAttSyntaxWarningIfNeeded(data, 'intel');
            expect(result.url).toBe(mockUrl);
        });
    });
});

describe('determineAssemblySyntax', () => {
    it.each([
        ['intel', undefined, false],
        ['intel', undefined, true],
        ['intel', false, false],
        ['intel', false, true],
        ['att', true, false],
        ['intel', true, true],
    ])('returns %s when supportsIntel is %s and intel asm syntax filter is %s', (expected, supportsIntel, intelFilterEnabled) => {
        expect(determineAssemblySyntax(supportsIntel, intelFilterEnabled)).toBe(expected);
    });
});
