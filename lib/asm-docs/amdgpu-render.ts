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

// Renders the normalised spec emitted by etc/scripts/docenizers/docenizer-amdgpu.py.
// Encoding, format and operand descriptions recur across thousands of instructions, so the
// generated files pool them and reference them by index; this expands that back to markup.

import type {AssemblyInstructionInfo} from '../../types/assembly-docs.interfaces.js';

/** Bit range of a single field within an encoding: [name, description, bitOffset, bitCount]. */
export type BitField = [string, string, number, number];

/** [description, bitCount, fields] */
export type EncodingMeta = [string, number, BitField[]];

/** [description, dataType] */
export type DataFormatMeta = [string, string];

/** [displayName, description] */
export type GroupMeta = [string, string];

/** Operand flag bits, matching the attributes on <Operand>. */
enum OperandFlag {
    Input = 1,
    Output = 2,
    Implicit = 4,
    BinaryMicrocodeRequired = 8,
}

/** Instruction flag bits, matching the children of <InstructionFlags>. */
const INSTRUCTION_FLAG_LABELS: readonly string[] = [
    'Branch',
    'Conditional Branch',
    'Indirect Branch',
    'Program Terminator',
    'Immediately Executed',
];

/** [fieldName, operandTypeName, dataFormatName, sizeInBits, flags] */
export type OperandRow = [string, string, string, number, number];

/** [encodingName, condition, conditionId, operandIndices] */
export type EncShape = [string, string, string, number[]];

/** [name, description, aliases, groupName, subgroups, flags, [shapeIndex, opcode][]] */
export type InstrRow = [string, string, string[], string, string[], number, [number, number][]];

export interface AmdIsaSpec {
    url: string;
    encodings: Record<string, EncodingMeta>;
    formats: Record<string, DataFormatMeta>;
    operandTypes: Record<string, string>;
    groups: Record<string, GroupMeta>;
    /** Pooled operand rows, referenced by index from `shapes`. */
    operands: OperandRow[];
    /** Pooled encoding shapes, referenced by index from `instructions`. */
    shapes: EncShape[];
    instructions: InstrRow[];
    /** Lowercased instruction names and aliases, mapped to an index into `instructions`. */
    index: Record<string, number>;
    /** Encoding suffix (`e32`, `e64`, ...) to the encoding names it selects. */
    suffixEncodings: Record<string, string[]>;
}

// Suffixes llvm-objdump appends to disambiguate an instruction's forms. Longest first, so
// `_e64_dpp` is not mistaken for `_dpp`.
const ENCODING_SUFFIXES: readonly string[] = [
    '_e64_dpp',
    '_e32_dpp',
    '_dpp16',
    '_dpp8',
    '_sdwa',
    '_dpp',
    '_e64',
    '_e32',
];

/** Suffix aliases: `_dpp` on its own means the 16-lane form. */
const SUFFIX_KEYS: Record<string, string> = {
    e32: 'e32',
    e64: 'e64',
    dpp: 'dpp16',
    dpp8: 'dpp8',
    dpp16: 'dpp16',
    sdwa: 'sdwa',
    e32_dpp: 'dpp16',
    e64_dpp: 'dpp16',
};

const HTML_ESCAPES: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
};

function escapeHtml(text: string): string {
    return text.replace(/[&<>"]/g, c => HTML_ESCAPES[c]);
}

const TABLE_STYLE = 'border-collapse:separate;border-spacing:14px 2px;margin:2px 0 6px 4px';

function field(label: string, value: string): string {
    return `<p><b>${label}:</b> ${escapeHtml(value)}</p>`;
}

function table(headers: readonly string[], rows: string[][]): string {
    const head = headers.map(h => `<th align="left">${h}</th>`).join('');
    const body = rows.map(cells => `<tr>${cells.map(c => `<td>${c}</td>`).join('')}</tr>`).join('');
    return `<table style="${TABLE_STYLE}"><tr>${head}</tr>${body}</table>`;
}

/** `OPR_VGPR` -> `VGPR`, `FMT_NUM_F32` -> `f32`. The full name stays in the cell's title. */
function shortenIdentifier(name: string): string {
    if (name.startsWith('OPR_')) return name.slice(4);
    if (name.startsWith('FMT_NUM_')) return name.slice(8).toLowerCase();
    if (name.startsWith('FMT_')) return name.slice(4).toLowerCase();
    return name;
}

function cellWithTitle(name: string, title: string): string {
    const shortened = escapeHtml(shortenIdentifier(name));
    return title ? `<span title="${escapeHtml(`${name}: ${title}`)}">${shortened}</span>` : shortened;
}

// Conditions arrive as run-on names: `Nothas_lit_0_Nothas_lit_1`, `has_dpp8`. Split them and
// drop the has/Nothas prefixes. The `_N` suffixes are deliberately left alone rather than
// read as source indices -- ENC_SOP1 carries Nothas_lit_1 with only one source operand.
function describeCondition(condition: string): string {
    if (!condition || condition === 'default') return '';
    return condition
        .split(/_(?=has_|Nothas_)/)
        .map(part =>
            part.startsWith('Nothas_') ? `no ${part.slice(7)}` : part.startsWith('has_') ? part.slice(4) : part,
        )
        .join(', ');
}

function describeOperandDirection(flags: number): string {
    const isInput = (flags & OperandFlag.Input) !== 0;
    const isOutput = (flags & OperandFlag.Output) !== 0;
    if (isInput && isOutput) return 'in/out';
    if (isOutput) return 'out';
    if (isInput) return 'in';
    return '';
}

function renderEncoding(spec: AmdIsaSpec, shapeIndex: number, opcode: number, isActive: boolean): string {
    const [encodingName, condition, , operandIndices] = spec.shapes[shapeIndex];
    const meta = spec.encodings[encodingName];
    const parts: string[] = [];

    const summaryDetails: string[] = [];
    const conditionText = describeCondition(condition);
    if (conditionText) summaryDetails.push(escapeHtml(conditionText));
    summaryDetails.push(`opcode: ${opcode}`);
    if (meta && meta[1]) summaryDetails.push(`${meta[1]}-bit`);
    const activeMark = isActive ? ' <b>&larr; active</b>' : '';
    parts.push(
        `<summary><code>${escapeHtml(encodingName)}</code>` +
            ` <i>(${summaryDetails.join(', ')})</i>${activeMark}</summary>`,
    );

    if (meta?.[0]) parts.push(`<p style="margin:2px 0 4px 8px"><i>${escapeHtml(meta[0])}</i></p>`);

    if (operandIndices.length > 0) {
        const rows = operandIndices.map((operandIndex, position) => {
            const [fieldName, typeName, formatName, size, flags] = spec.operands[operandIndex];
            const format = spec.formats[formatName];
            // The data type (float, integer, ...) is only in the spec data, so surface it here.
            const formatTitle = format ? (format[1] ? `(${format[1]}) ${format[0]}` : format[0]) : '';
            return [
                String(position + 1),
                `<code>${escapeHtml(fieldName || '(implicit)')}</code>` +
                    (flags & OperandFlag.Implicit ? ' <i>(implicit)</i>' : ''),
                cellWithTitle(typeName, spec.operandTypes[typeName] ?? ''),
                cellWithTitle(formatName, formatTitle),
                `${size}b`,
                describeOperandDirection(flags),
            ];
        });
        parts.push(table(['#', 'Operand', 'Type', 'Format', 'Size', 'Dir'], rows));
    }

    const bitFields = meta?.[2];
    if (bitFields && bitFields.length > 0) {
        // Sorted high bit first so the layout reads like a bit diagram; the spec's own order
        // is whatever the XML listed.
        const rows = [...bitFields]
            .sort((a, b) => b[2] - a[2])
            .map(([name, description, offset, count]) => [
                `<code>${escapeHtml(name)}</code>`,
                `<span style="white-space:nowrap">[${offset + count - 1}:${offset}]</span>`,
                String(count),
                escapeHtml(description),
            ]);
        parts.push(`<p style="margin:4px 0 0 8px"><b>Bit layout</b></p>`);
        parts.push(table(['Field', 'Bits', 'Width', 'Description'], rows));
    }

    return `<details style="margin:0 0 2px 8px">${parts.join('\n')}</details>`;
}

function buildTooltip(instruction: InstrRow): string {
    const [name, description] = instruction;
    const parts = [`Instruction: ${name}`];
    if (description) {
        // Rendered as markdown, where a lone newline collapses; only a blank line breaks the paragraph.
        parts.push(`Description: ${description.length > 200 ? `${description.slice(0, 197)}...` : description}`);
    }
    return parts.join('\n\n');
}

function buildHtml(spec: AmdIsaSpec, instruction: InstrRow, activeEncodings: readonly string[]): string {
    const [name, description, aliases, groupName, subgroups, flags, encodings] = instruction;
    const parts: string[] = [field('Instruction', name)];

    if (aliases.length > 0) parts.push(field('Aliases', aliases.join(', ')));
    if (description) parts.push(field('Description', description));

    const group = spec.groups[groupName];
    if (group) {
        parts.push(field('Functional Group', group[0]));
        if (group[1]) parts.push(`<p style="margin:0 0 12px 8px"><i>${escapeHtml(group[1])}</i></p>`);
    }
    if (subgroups.length > 0) parts.push(field('Functional Sub-Groups', subgroups.join(', ')));

    const flagLabels = INSTRUCTION_FLAG_LABELS.filter((_, bit) => (flags & (1 << bit)) !== 0);
    if (flagLabels.length > 0) parts.push(field('Flags', flagLabels.join(', ')));

    if (encodings.length > 0) {
        // The suffix tells us which form is live, so lead with it.
        const isActive = (shapeIndex: number) => activeEncodings.includes(spec.shapes[shapeIndex][0]);
        const ordered =
            activeEncodings.length > 0 ? [...encodings].sort((a, b) => +isActive(b[0]) - +isActive(a[0])) : encodings;

        parts.push(`<p style="margin-bottom:2px"><b>Encodings:</b> (${encodings.length})</p>`);
        for (const [shapeIndex, opcode] of ordered) {
            parts.push(renderEncoding(spec, shapeIndex, opcode, isActive(shapeIndex)));
        }
    }

    // No trailing link: appendInfo in static/panes/compiler.ts already adds one from `url`.
    return parts.join('\n');
}

/** Falls back to stripping an encoding suffix, reporting which encodings it selected. */
function resolve(spec: AmdIsaSpec, opcode: string): {index: number; activeEncodings: readonly string[]} | undefined {
    const exact = spec.index[opcode];
    if (exact !== undefined) return {index: exact, activeEncodings: []};

    for (const suffix of ENCODING_SUFFIXES) {
        if (!opcode.endsWith(suffix)) continue;
        const index = spec.index[opcode.slice(0, -suffix.length)];
        // Keep looking rather than bailing: v_add_f32_e64_dpp must fall through to _dpp.
        if (index === undefined) continue;
        return {index, activeEncodings: spec.suffixEncodings[SUFFIX_KEYS[suffix.slice(1)]] ?? []};
    }
    return undefined;
}

export function buildAsmDocs(spec: AmdIsaSpec, opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    const resolved = resolve(spec, opcode.toLowerCase());
    if (!resolved) return;
    const instruction = spec.instructions[resolved.index];
    return {
        html: buildHtml(spec, instruction, resolved.activeEncodings),
        tooltip: buildTooltip(instruction),
        url: spec.url,
    };
}
