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

// The generator pools operands and encodings and references them by index, so a
// mis-association fails silently - the output stays well-formed but describes the wrong
// instruction. Hence walking the whole dataset rather than sampling it.
//
// Fidelity to the source XML is checked by `docenizer-amdgpu.py --verify`; the XML is not
// committed.

import {describe, expect, it} from 'vitest';

import type {AmdIsaSpec} from '../lib/asm-docs/amdgpu-render.js';
import * as cdna1 from '../lib/asm-docs/generated/asm-docs-amd_cdna1.js';
import * as cdna2 from '../lib/asm-docs/generated/asm-docs-amd_cdna2.js';
import * as cdna3 from '../lib/asm-docs/generated/asm-docs-amd_cdna3.js';
import * as cdna4 from '../lib/asm-docs/generated/asm-docs-amd_cdna4.js';
import * as cdna5 from '../lib/asm-docs/generated/asm-docs-amd_cdna5.js';
import * as rdna1 from '../lib/asm-docs/generated/asm-docs-amd_rdna1.js';
import * as rdna2 from '../lib/asm-docs/generated/asm-docs-amd_rdna2.js';
import * as rdna3 from '../lib/asm-docs/generated/asm-docs-amd_rdna3.js';
import * as rdna3_5 from '../lib/asm-docs/generated/asm-docs-amd_rdna3_5.js';
import * as rdna4 from '../lib/asm-docs/generated/asm-docs-amd_rdna4.js';
import {getAmdGpuInstructionSet} from '../lib/instructionsets.js';
import {skipExpensiveTests} from './utils.js';

type GeneratedModule = {
    SPEC: AmdIsaSpec;
    getAsmOpcode: (opcode: string | undefined) => {html: string; tooltip: string; url: string} | undefined;
};

const ARCHITECTURES: [string, GeneratedModule][] = [
    ['amd_cdna1', cdna1],
    ['amd_cdna2', cdna2],
    ['amd_cdna3', cdna3],
    ['amd_cdna4', cdna4],
    ['amd_cdna5', cdna5],
    ['amd_rdna1', rdna1],
    ['amd_rdna2', rdna2],
    ['amd_rdna3', rdna3],
    ['amd_rdna3_5', rdna3_5],
    ['amd_rdna4', rdna4],
];

// Instruction record layout, mirroring InstrRow in amdgpu-render.ts.
const NAME = 0;
const ALIASES = 2;
const GROUP = 3;
const ENCODINGS = 6;

function firstFew(problems: string[], limit = 10): string[] {
    return problems.length > limit
        ? [...problems.slice(0, limit), `... and ${problems.length - limit} more`]
        : problems;
}

/** Every tag amdgpu-render.ts emits. Each contributes exactly one `<` and one `>`. */
const EMITTED_TAG = /<\/?(?:p|b|i|code|span|table|tr|th|td|details|summary)(?: [^<>]*)?>/g;

/**
 * True when the output holds an angle bracket that no emitted tag accounts for, which is what
 * an unescaped description would leave behind. Counted rather than stripped: removing tags with
 * a regex is unreliable, since a nested `<scr<span>ipt>` survives the pass.
 */
function hasStrayAngleBracket(html: string): boolean {
    const tags = (html.match(EMITTED_TAG) ?? []).length;
    let open = 0;
    let close = 0;
    for (const character of html) {
        if (character === '<') open++;
        else if (character === '>') close++;
    }
    return open !== tags || close !== tags;
}

describe('AMD GPU asm-docs generated data', () => {
    it('loads a non-empty spec for every architecture', () => {
        expect(ARCHITECTURES).toHaveLength(10);
        for (const [arch, mod] of ARCHITECTURES) {
            expect(mod.SPEC.instructions.length, `${arch} instructions`).toBeGreaterThan(0);
            expect(Object.keys(mod.SPEC.index).length, `${arch} index`).toBeGreaterThanOrEqual(
                mod.SPEC.instructions.length,
            );
            expect(mod.SPEC.url, `${arch} url`).toMatch(/^https:\/\//);
        }
    });

    it.each(ARCHITECTURES)('%s has no dangling pool references', (arch, mod) => {
        const spec = mod.SPEC;
        const problems: string[] = [];
        for (const instruction of spec.instructions) {
            for (const [shapeIndex] of instruction[ENCODINGS]) {
                const shape = spec.shapes[shapeIndex];
                if (shape === undefined) {
                    problems.push(`${instruction[NAME]}: shape index ${shapeIndex} out of range`);
                    continue;
                }
                for (const operandIndex of shape[3]) {
                    if (spec.operands[operandIndex] === undefined) {
                        problems.push(`${instruction[NAME]}/${shape[0]}: operand index ${operandIndex} out of range`);
                    }
                }
            }
        }
        expect(firstFew(problems), `${arch} dangling references`).toEqual([]);
    });

    it.each(ARCHITECTURES)('%s only references names present in the lookup tables', (arch, mod) => {
        const spec = mod.SPEC;
        const problems: string[] = [];
        for (const instruction of spec.instructions) {
            const group = instruction[GROUP];
            if (group && !(group in spec.groups)) problems.push(`${instruction[NAME]}: unknown group ${group}`);
            for (const [shapeIndex] of instruction[ENCODINGS]) {
                const shape = spec.shapes[shapeIndex];
                if (!(shape[0] in spec.encodings)) problems.push(`${instruction[NAME]}: unknown encoding ${shape[0]}`);
                for (const operandIndex of shape[3]) {
                    const [, typeName, formatName] = spec.operands[operandIndex];
                    if (typeName && !(typeName in spec.operandTypes)) {
                        problems.push(`${instruction[NAME]}: unknown operand type ${typeName}`);
                    }
                    if (formatName && !(formatName in spec.formats)) {
                        problems.push(`${instruction[NAME]}: unknown data format ${formatName}`);
                    }
                }
            }
        }
        expect(firstFew(problems), `${arch} unknown names`).toEqual([]);
    });

    it.each(ARCHITECTURES)('%s index resolves every instruction and alias', (arch, mod) => {
        const spec = mod.SPEC;
        const problems: string[] = [];

        for (const [position, instruction] of spec.instructions.entries()) {
            if (spec.index[instruction[NAME].toLowerCase()] !== position) {
                problems.push(`${instruction[NAME]}: not reachable by its own name`);
            }
            for (const alias of instruction[ALIASES]) {
                if (spec.index[alias.toLowerCase()] === undefined) {
                    problems.push(`${instruction[NAME]}: alias ${alias} missing from index`);
                }
            }
        }

        // Every key must land on an instruction that actually claims that name.
        for (const [key, position] of Object.entries(spec.index)) {
            const instruction = spec.instructions[position];
            if (instruction === undefined) {
                problems.push(`${key}: index points outside instructions`);
                continue;
            }
            const claimed = [instruction[NAME], ...instruction[ALIASES]].map(n => n.toLowerCase());
            if (!claimed.includes(key))
                problems.push(`${key}: resolves to ${instruction[NAME]}, which does not claim it`);
        }

        expect(firstFew(problems), `${arch} index`).toEqual([]);
    });
});

describe('AMD GPU target mapping', () => {
    it('covers both RDNA and CDNA families', () => {
        const expected: [string, string][] = [
            ['gfx1010', 'amd_rdna1'],
            ['gfx1030', 'amd_rdna2'],
            ['gfx1100', 'amd_rdna3'],
            ['gfx1151', 'amd_rdna3_5'],
            ['gfx1200', 'amd_rdna4'],
            ['gfx908', 'amd_cdna1'],
            ['gfx90a', 'amd_cdna2'],
            ['gfx942', 'amd_cdna3'],
            ['gfx950', 'amd_cdna4'],
            ['gfx1250', 'amd_cdna5'],
        ];
        for (const [target, instructionSet] of expected) {
            expect(getAmdGpuInstructionSet(target), target).toEqual(instructionSet);
        }
    });

    it('prefers the more specific prefix where two overlap', () => {
        // gfx125x is CDNA5 but starts with gfx12 (RDNA4); gfx115x is RDNA3.5 but starts with gfx11.
        // Whole families, not single revisions: gfx1251 is as real a CDNA5 target as gfx1250.
        for (const target of ['gfx1250', 'gfx1251']) {
            expect(getAmdGpuInstructionSet(target), target).toEqual('amd_cdna5');
        }
        for (const target of ['gfx1150', 'gfx1151', 'gfx1152', 'gfx1153']) {
            expect(getAmdGpuInstructionSet(target), target).toEqual('amd_rdna3_5');
        }
    });

    it('is case insensitive', () => {
        expect(getAmdGpuInstructionSet('GFX1100')).toEqual('amd_rdna3');
    });

    it('returns undefined for targets we ship no docs for', () => {
        for (const target of ['gfx900', 'gfx906', 'sm_80', '', 'gfx']) {
            expect(getAmdGpuInstructionSet(target), target).toBeUndefined();
        }
    });
});

describe('AMD GPU asm-docs lookup behaviour', () => {
    it('resolves a legacy GCN alias to its canonical instruction', () => {
        const info = rdna4.getAsmOpcode('s_load_dword');
        expect(info?.tooltip).toContain('Instruction: S_LOAD_B32');
        expect(info?.html).toContain('S_LOAD_DWORD');
    });

    it('resolves an encoding suffix and marks the encoding it selects', () => {
        const wide = rdna4.getAsmOpcode('v_add_f32_e64');
        expect(wide?.tooltip).toContain('Instruction: V_ADD_F32');
        expect(wide?.html).toMatch(/<summary><code>ENC_VOP3<\/code>[^<]*<i>[^<]*<\/i> <b>&larr; active<\/b>/);

        const narrow = rdna4.getAsmOpcode('v_add_f32_e32');
        expect(narrow?.html).toMatch(/<summary><code>ENC_VOP2<\/code>[^<]*<i>[^<]*<\/i> <b>&larr; active<\/b>/);
    });

    it('keeps every encoding when one is marked active', () => {
        const count = (html: string) => (html.match(/<summary>/g) ?? []).length;
        const bare = rdna4.getAsmOpcode('v_add_f32')!;
        expect(count(bare.html)).toBeGreaterThan(1);
        expect(count(rdna4.getAsmOpcode('v_add_f32_e32')!.html)).toEqual(count(bare.html));
        expect(count(rdna4.getAsmOpcode('v_add_f32_e64')!.html)).toEqual(count(bare.html));
    });

    it('collapses each encoding behind its own summary', () => {
        const html = rdna4.getAsmOpcode('v_add_f32')!.html;
        expect(html).toContain('<b>Encodings:</b> (10)');
        // Nothing is pre-expanded: a stray `open` would undo the point of collapsing.
        expect(html).not.toMatch(/<details[^>]*\bopen\b/);
    });

    it('shortens spec identifiers but keeps the original in the hover title', () => {
        const html = rdna4.getAsmOpcode('v_add_f32_e32')!.html;
        expect(html).toContain('<span title="OPR_VGPR: Operand must be a vector GPR');
        expect(html).toMatch(/>VGPR<\/span>/);
        // The data type from the spec rides along in the format title.
        expect(html).toContain('FMT_NUM_F32: (float)');
        expect(html).toMatch(/>f32<\/span>/);
    });

    it('drops the fields that carry no signal', () => {
        // bmr is set on 438 of rdna4's 457 operand rows, so it never distinguishes anything.
        const html = rdna4.getAsmOpcode('v_add_f32_e32')!.html;
        expect(html).not.toContain('bmr');
        expect(html).not.toContain('condition: default');
        // S_LOAD_B32 has no subgroups; the field should be absent rather than "N/A".
        expect(rdna4.getAsmOpcode('s_load_dword')!.html).not.toContain('N/A');
    });

    it('humanises encoding conditions without inventing operand indices', () => {
        const html = rdna4.getAsmOpcode('s_mov_b32')!.html;
        expect(html).toContain('no lit_0, no lit_1');
        expect(html).not.toContain('Nothas_');
    });

    it('orders bit layout rows from the high bit down', () => {
        const html = rdna4.getAsmOpcode('v_add_f32_e32')!.html;
        // Each encoding carries its own layout, so the offsets restart; check per encoding.
        const blocks = html.split('<details').slice(1);
        expect(blocks.length).toBeGreaterThan(1);
        for (const block of blocks) {
            // Match the Bits cell specifically: descriptions quote bit ranges in prose too.
            const offsets = [...block.matchAll(/white-space:nowrap">\[\d+:(\d+)]/g)].map(m => Number(m[1]));
            expect(offsets.length, 'encoding should have a bit layout').toBeGreaterThan(1);
            expect(
                [...offsets].sort((a, b) => b - a),
                block.slice(0, 80),
            ).toEqual(offsets);
        }
    });

    it('does not invent matches for unknown opcodes or bogus suffixes', () => {
        for (const opcode of ['not_an_opcode', 'v_add_f32_e99', '_e32', '', undefined]) {
            expect(rdna4.getAsmOpcode(opcode), `${opcode} should not resolve`).toBeUndefined();
        }
    });

    it('escapes markup characters coming from the ISA descriptions', () => {
        const spec = rdna4.SPEC;
        const risky = spec.instructions.find(instruction => /[<>&]/.test(instruction[1]));
        expect(risky, 'expected at least one description containing markup characters').toBeDefined();
        const html = rdna4.getAsmOpcode(risky![NAME].toLowerCase())!.html;
        // The description must appear escaped, and never in its raw form.
        const description = risky![1];
        // Mirrors escapeHtml in amdgpu-render.ts, including the quote it escapes for attributes.
        const escaped = description
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
        expect(html).toContain(escaped);
        expect(escaped, 'test needs a description that actually changes when escaped').not.toEqual(description);
        expect(html).not.toContain(description);
        expect(hasStrayAngleBracket(html)).toBe(false);
    });
});

// ~13k renders, so gated out of pre-commit, which runs test-min.
describe.skipIf(skipExpensiveTests)('AMD GPU asm-docs render sweep', () => {
    it.each(ARCHITECTURES)('%s renders every indexed key', (arch, mod) => {
        const problems: string[] = [];
        for (const key of Object.keys(mod.SPEC.index)) {
            let info: ReturnType<GeneratedModule['getAsmOpcode']>;
            try {
                info = mod.getAsmOpcode(key);
            } catch (error) {
                problems.push(`${key}: threw ${error}`);
                continue;
            }
            if (!info) {
                problems.push(`${key}: indexed but did not resolve`);
                continue;
            }
            if (!info.tooltip.startsWith('Instruction: ')) problems.push(`${key}: tooltip malformed`);
            if (!info.html.startsWith('<p><b>Instruction:</b> ')) problems.push(`${key}: html malformed`);
            // Structural markers only: "NaN" is legitimate prose in v_div_fixup_*.
            if (/>undefined<|opcode: undefined|\[object |undefined-bit/.test(info.html)) {
                problems.push(`${key}: failed lookup leaked into html`);
            }
            if (hasStrayAngleBracket(info.html)) {
                problems.push(`${key}: unescaped markup`);
            }
        }
        expect(firstFew(problems), `${arch} render sweep`).toEqual([]);
    });
});
