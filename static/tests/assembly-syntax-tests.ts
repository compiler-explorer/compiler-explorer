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
import {ATT_SYNTAX_WARNING, addAttSyntaxWarningIfNeeded} from '../assembly-syntax.js';

function makeInfo(tooltip: string, html: string): AssemblyInstructionInfo {
    return {tooltip, html, url: 'https://example.com'};
}

describe('addAttSyntaxWarningIfNeeded', () => {
    it('does not warn for intel syntax', () => {
        const data = makeInfo('first operand is the destination', 'first operand is the destination');
        const result = addAttSyntaxWarningIfNeeded(data, 'intel');
        expect(result).toEqual(data);
    });

    it('does not warn for att syntax when no cardinality references exist', () => {
        const data = makeInfo('Adds two values', '<p>Adds two values</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
        expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
    });

    it('does not warn when cardinality is present but source/destination is absent', () => {
        const data = makeInfo('The first operand is added to the result', '<p>The first operand is added</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
    });

    it('does not warn when source/destination is present but cardinality is absent', () => {
        const data = makeInfo('Copies from source to destination', '<p>Copies source to destination</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
    });

    it('warns when tooltip references cardinality operand and source/destination', () => {
        const data = makeInfo('The first operand is the destination register', '<p>Simple description</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
        expect(result.html).toContain(ATT_SYNTAX_WARNING);
    });

    it('warns when html references cardinality operand and source/destination', () => {
        const data = makeInfo('Simple tooltip', '<p>The second operand is the source register</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
        expect(result.html).toContain(ATT_SYNTAX_WARNING);
    });

    it('warns when both tooltip and html match', () => {
        const text = 'The first operand is the source, the second operand is the destination';
        const data = makeInfo(text, text);
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
        expect(result.html).toContain(ATT_SYNTAX_WARNING);
    });

    it('does not cross-contaminate: tooltip cardinality + html source/dest alone should not warn', () => {
        const data = makeInfo('The first operand is shifted left', '<p>Copies data to the destination register</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
        expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
    });

    it('handles fourth operand references', () => {
        const data = makeInfo('The fourth operand specifies the destination mask', '<p>VBLENDVPS description</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
    });

    it('handles last operand references', () => {
        const data = makeInfo('The last operand is the source register', '<p>Description</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
    });

    it('does not mutate the original data object', () => {
        const data = makeInfo('The first operand is the destination', '<p>The first operand is the destination</p>');
        const originalTooltip = data.tooltip;
        const originalHtml = data.html;
        addAttSyntaxWarningIfNeeded(data, 'att');
        expect(data.tooltip).toBe(originalTooltip);
        expect(data.html).toBe(originalHtml);
    });

    it('preserves the url field unchanged', () => {
        const data = makeInfo('The first operand is the destination', '<p>The first operand is the destination</p>');
        const result = addAttSyntaxWarningIfNeeded(data, 'att');
        expect(result.url).toBe('https://example.com');
    });
});
