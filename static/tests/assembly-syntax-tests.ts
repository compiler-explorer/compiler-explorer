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

import { describe, expect, it } from "vitest";

import { type AssemblyInstructionInfo } from "../../types/assembly-docs.interfaces.js";
import {
    ATT_SYNTAX_WARNING,
    addAttSyntaxWarningIfNeeded,
    determineAssemblySyntax,
} from "../assembly-syntax.js";

function makeInfo(tooltip: string, html?: string): AssemblyInstructionInfo {
    html = html ?? `<p>${tooltip}</p>`;
    return { tooltip, html, url: "https://example.com" };
}

describe(addAttSyntaxWarningIfNeeded, () => {
    describe("warns when syntax is att and", () => {
        it("tooltip references cardinality operand and source/destination", () => {
            const data = makeInfo(
                "first operand (destination)",
                "<p>Simple description</p>",
            );
            const result = addAttSyntaxWarningIfNeeded(data, "att");

            expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
            expect(result.html).toContain(ATT_SYNTAX_WARNING);
        });

        it("html references cardinality operand and source/destination", () => {
            const data = makeInfo(
                "Simple tooltip",
                "<p>second operand (source)</p>",
            );
            const result = addAttSyntaxWarningIfNeeded(data, "att");

            expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
            expect(result.html).toContain(ATT_SYNTAX_WARNING);
        });

        it("both tooltip and html reference cardinality and source/destination", () => {
            const text =
                "copies the first operand (source) to the second operand (destination)";
            const data = makeInfo(text);
            const result = addAttSyntaxWarningIfNeeded(data, "att");
            expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
            expect(result.html).toContain(ATT_SYNTAX_WARNING);
        });
    });
    describe("does not warn when", () => {
        it("intel syntax and references cardinal destination", () => {
            const data = makeInfo("first operand (destination)");
            const result = addAttSyntaxWarningIfNeeded(data, "intel");
            expect(result).toEqual(data);
        });

        it("att syntax and no cardinality references exist", () => {
            const data = makeInfo(
                "Decrements the stack pointer and then stores the source operand on the top of the stack",
            );
            const result = addAttSyntaxWarningIfNeeded(data, "att");

            expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
            expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
        });

        it("cardinality is present but source/destination is absent", () => {
            const data = makeInfo(
                "Performs a signed multiplication of two operands",
            );
            const result = addAttSyntaxWarningIfNeeded(data, "att");

            expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
        });

        it("source/destination is present but cardinality is absent", () => {
            const data = makeInfo(
                "Loads the value from the top of the stack to the location specified with the destination operand (or explicit opcode) and then increments the stack pointer.",
            );
            const result = addAttSyntaxWarningIfNeeded(data, "att");
            expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
        });
    });

    it("does not cross-contaminate: tooltip cardinality + html source/dest alone should not warn", () => {
        const data = makeInfo(
            "The first operand is foo",
            "<p>Copies bar to the destination register</p>",
        );
        const result = addAttSyntaxWarningIfNeeded(data, "att");
        expect(result.tooltip).not.toContain(ATT_SYNTAX_WARNING);
        expect(result.html).not.toContain(ATT_SYNTAX_WARNING);
    });

    it("handles fourth operand references", () => {
        const data = makeInfo(
            "The fourth operand specifies the destination mask",
            "<p>VBLENDVPS description</p>",
        );
        const result = addAttSyntaxWarningIfNeeded(data, "att");
        expect(result.tooltip).toContain(ATT_SYNTAX_WARNING);
    });

    describe("purity", () => {
        it("does not mutate the original data object regardless of syntax", () => {
            const data = makeInfo(
                "The first operand is the destination",
                "<p>The first operand is the destination</p>",
            );
            const originalTooltip = data.tooltip;
            const originalHtml = data.html;

            addAttSyntaxWarningIfNeeded(data, "att");
            expect(data.tooltip).toBe(originalTooltip);
            expect(data.html).toBe(originalHtml);
        });

        it("preserves the url field regardless of syntax", () => {
            let result: AssemblyInstructionInfo;
            const data: AssemblyInstructionInfo = makeInfo(
                "The first operand (destination)",
                "<p>The first operand (destination)</p>",
            );

            result = addAttSyntaxWarningIfNeeded(data, "att");
            expect(result.url).toBe("https://example.com");
            result = addAttSyntaxWarningIfNeeded(data, "intel");
            expect(result.url).toBe("https://example.com");
        });
    });
});

describe('determineAssemblySyntax', () => {
    it.each([
        [false, false, 'intel'],
        [false, true, 'intel'],
        [true, false, 'att'],
        [true, true, 'intel'],
    ])('returns %s when supportsIntel is %s and intel asm syntax filter is %s', (supportsIntel, intelFilterEnabled, expected) => {
        expect(determineAssemblySyntax(supportsIntel as boolean, intelFilterEnabled as boolean)).toBe(expected);
    });
});
