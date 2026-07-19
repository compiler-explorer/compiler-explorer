// Copyright (c) 2026, Compiler Explorer Authors
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

// Delphi disassembly annotator, ported from the proven map2pdb + llvm-pdbutil pipeline in
// https://github.com/jimmckeeth/Delphi-Annotated-Disassembly.
//
// The compiler emits a Detailed .map (dcc -GD). map2pdb converts it to a PDB, and
// `llvm-pdbutil dump -l` (native reader; no msdia140.dll/DIA needed) yields per-source-line
// address entries plus each module's authoritative code boundary. We keep only the objdump
// instructions whose address falls inside a line-mapped range of the user's own source file,
// interleave source markers, and trim the post-routine data tail that objdump misdecodes -
// i.e. "only show what is traceable to the code".

export type LineEntry = {line: number; off: number};
export type Contribution = {
    file: string | null; // lowercased source-file basename
    seg: number;
    startOff: number;
    endOff: number;
    entries: LineEntry[];
};
export type AsmRange = {line: number; start: number; end: number; trailing: boolean};

export function basenameLower(p: string): string {
    const parts = p.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1].trim().toLowerCase();
}

// Parse the ".map" top segment table into { segNum -> virtual-address base }.
// Rows look like:  " 0001:00401000 0001F548H .text                   CODE"
// Parsing stops at the "Detailed map of segments" section (relative offsets, different format).
export function parseSegmentBases(mapText: string): Map<number, number> {
    const bases = new Map<number, number>();
    for (const line of mapText.split(/\r?\n/)) {
        if (/Detailed map of segments/i.test(line)) break;
        const m = line.match(/^\s*([0-9a-fA-F]{4}):([0-9a-fA-F]+)\s+[0-9a-fA-F]+H\s+\S/);
        if (m) bases.set(Number.parseInt(m[1], 16), Number.parseInt(m[2], 16));
    }
    return bases;
}

// Parse `llvm-pdbutil dump -l` output into contributions. Layout:
//   Mod 0021 | `Project32`:
//   Project32.dpr (no checksum)
//     0002:000009A0-00000C78, line/addr entries = 17
//       12 000009A0     19 000009D4   ...   (decimal source line, hex segment offset)
export function parseDumpLines(text: string): Contribution[] {
    const contribs: Contribution[] = [];
    let curFile: string | null = null;
    let curContrib: Contribution | null = null;
    for (const raw of text.split(/\r?\n/)) {
        const line = raw.replace(/\s+$/, '');
        if (/^Mod \d+ \|/.test(line)) {
            curContrib = null;
            continue;
        }
        const fm = line.match(/^(\S.*\.(?:pas|dpr|inc|p))\s+\(/i);
        if (fm) {
            curFile = basenameLower(fm[1]);
            curContrib = null;
            continue;
        }
        const cm = line.match(
            /^\s*([0-9a-fA-F]{4}):([0-9a-fA-F]+)-([0-9a-fA-F]+),\s*line\/addr entries\s*=\s*(\d+)/i,
        );
        if (cm) {
            curContrib = {
                file: curFile,
                seg: Number.parseInt(cm[1], 16),
                startOff: Number.parseInt(cm[2], 16),
                endOff: Number.parseInt(cm[3], 16),
                entries: [],
            };
            contribs.push(curContrib);
            continue;
        }
        if (curContrib && /^\s+\d+\s+[0-9a-fA-F]+/.test(line)) {
            const toks = line.trim().split(/\s+/);
            for (let i = 0; i + 1 < toks.length; i += 2) {
                const ln = Number.parseInt(toks[i], 10);
                const off = Number.parseInt(toks[i + 1], 16);
                if (!Number.isNaN(ln) && !Number.isNaN(off)) curContrib.entries.push({line: ln, off});
            }
        }
    }
    return contribs;
}

// Build sorted VA ranges for the target source file only. Each line entry spans to the next
// entry's address; the last entry of each contribution spans to the module boundary and is
// flagged `trailing` (that span also captures post-code data we trim in annotateAsm).
export function buildRanges(
    contribs: Contribution[],
    segBases: Map<number, number>,
    targetBasename: string,
): AsmRange[] {
    const target = targetBasename.toLowerCase();
    const ranges: AsmRange[] = [];
    for (const c of contribs) {
        if (c.file !== target) continue;
        const base = segBases.get(c.seg);
        if (base === undefined) continue;
        const endVA = base + c.endOff;
        const ents = c.entries.map(e => ({line: e.line, va: base + e.off})).sort((a, b) => a.va - b.va);
        for (let i = 0; i < ents.length; i++) {
            const start = ents[i].va;
            const isLast = i + 1 >= ents.length;
            const end = isLast ? endVA : ents[i + 1].va;
            if (end > start) ranges.push({line: ents[i].line, start, end, trailing: isLast});
        }
    }
    ranges.sort((a, b) => a.start - b.start);
    return ranges;
}

const INSTR_RE = /^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{2})\b/;
type Instr = {va: number; firstByte: string; isBad: boolean; hasMnemonic: boolean};
function parseInstr(line: string): Instr | null {
    const m = line.match(INSTR_RE);
    if (!m) return null;
    // objdump lays out "  <addr>:\t<hex bytes>\t<mnemonic ...>". A wide instruction's trailing
    // immediate bytes wrap onto a follow-on "  <addr>:\t<byte>" line with no mnemonic - those are
    // display noise, and their 00 bytes must not be mistaken for padding by trimTrailingEnd.
    const parts = line.split('\t');
    const hasMnemonic = parts.length >= 3 && parts[2].trim().length > 0;
    return {
        va: Number.parseInt(m[1], 16),
        firstByte: m[2].toLowerCase(),
        isBad: /\(bad\)/.test(line),
        hasMnemonic,
    };
}

// A trailing range balloons past the routine's real code into padding/data that objdump
// misdecodes. Real code ends at the first real (mnemonic-bearing) `(bad)`, int3 (cc) or 00-padding
// instruction at/after the range start; everything from there on is padding/data/other routines.
function trimTrailingEnd(asmLines: string[], start: number, end: number): number {
    for (const line of asmLines) {
        const ins = parseInstr(line);
        if (!ins || ins.va < start) continue;
        if (ins.va >= end) break;
        if (!ins.hasMnemonic) continue;
        if (ins.isBad || ins.firstByte === '00' || ins.firstByte === 'cc') return ins.va;
    }
    return end;
}

export const NOTE_NO_MAP =
    '// (delphi) no source-mapped code found for this unit - richer fallback is planned future work';

// Turn raw objdump lines into CE-binary-parser-ready lines: one user-function label, per-source
// line markers (Windows lineRe form: X:/path:line), and only instructions inside a kept range.
// Trailing ranges are clamped to where real code ends; misdecoded `(bad)` lines are dropped.
export function annotateAsm(
    asmLines: string[],
    ranges: AsmRange[],
    markerPath: string,
    labelName: string,
): string[] {
    if (!ranges.length) return [NOTE_NO_MAP];
    const effRanges = ranges.map(r =>
        r.trailing ? {...r, end: trimTrailingEnd(asmLines, r.start, r.end)} : r,
    );
    const out: string[] = [];
    let lastLine: number | null = null;
    let labelEmitted = false;
    const findLine = (va: number): number | null => {
        for (const r of effRanges) if (va >= r.start && va < r.end) return r.line;
        return null;
    };
    for (const line of asmLines) {
        const ins = parseInstr(line);
        if (!ins) continue;
        const ln = findLine(ins.va);
        if (ln === null) continue;
        if (ins.isBad || !ins.hasMnemonic) continue;
        if (!labelEmitted) {
            out.push(ins.va.toString(16).padStart(8, '0') + ' <' + labelName + '>:');
            labelEmitted = true;
        }
        if (ln !== lastLine) {
            out.push(markerPath + ':' + ln);
            lastLine = ln;
        }
        out.push(line);
    }
    return out.length ? out : [NOTE_NO_MAP];
}
