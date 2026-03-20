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
