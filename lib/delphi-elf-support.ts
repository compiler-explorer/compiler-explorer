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

// Delphi *Linux* (dcclinux64) disassembly annotator. Unlike the Windows target, the ELF carries DWARF
// debug info, so `objdump -dl` already interleaves `<path>:<line>` source markers. We keep only the
// instructions under the user's own source file (the RTL carries no user source markers), rewrite the
// markers to CE's Windows lineRe form (`X:/path:line`), and drop the trailing int3 padding. No map2pdb
// needed here - much simpler than the Windows PDB pipeline.

export function basenameLower(p: string): string {
    const parts = p.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1].trim().toLowerCase();
}

const LABEL_RE = /^[0-9a-fA-F]+\s+<(.+)>:$/;
const SRC_RE = /^(.+\.(?:pas|dpr|inc|p)):(\d+)$/i;
const INSTR_RE = /^\s*([0-9a-fA-F]+):\s/;
const NOTE = '// (delphi) no source-mapped code found for this unit - richer fallback is planned future work';

export function annotateElfAsm(asmLines: string[], targetBasename: string, labelName: string): string[] {
    const target = targetBasename.toLowerCase();
    const out: string[] = [];
    let keep = false;
    let currentLine: number | null = null;
    let lastEmitted: number | null = null;
    let labelEmitted = false;
    for (const raw of asmLines) {
        const line = raw.replace(/\s+$/, '');
        if (LABEL_RE.test(line)) {
            keep = false; // entering a new function; re-enabled only by a user source marker
            continue;
        }
        const sm = line.match(SRC_RE);
        if (sm) {
            if (basenameLower(sm[1]) === target) {
                keep = true;
                currentLine = Number.parseInt(sm[2], 10);
            } else {
                keep = false;
                currentLine = null;
            }
            continue;
        }
        const im = line.match(INSTR_RE);
        if (im && keep) {
            const parts = line.split('\t');
            const hasMnemonic = parts.length >= 3 && parts[2].trim().length > 0;
            if (!hasMnemonic) continue; // objdump byte-continuation line
            if (!labelEmitted) {
                out.push(im[1].padStart(8, '0') + ' <' + labelName + '>:');
                labelEmitted = true;
            }
            if (currentLine !== lastEmitted) {
                out.push('C:/app/' + targetBasename + ':' + currentLine);
                lastEmitted = currentLine;
            }
            out.push(line);
        }
    }
    // trim trailing int3 (0xcc) padding after the routine's real code
    while (out.length > 0 && /\bint3\b/.test(out[out.length - 1])) out.pop();
    return out.length > 0 ? out : [NOTE];
}
