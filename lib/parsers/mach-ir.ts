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

import type {AsmResultSource, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import * as utils from '../utils.js';

const debugOpenRe = /^\s*debug\s*\{\s*$/;
const debugDirRe = /^(\s*directory\s*=\s*")((?:[^"\\]|\\.)*)(";\s*)$/;
const debugFileRe = /^(\s*file\s+([A-Za-z_]\w*)\s*=\s*")((?:[^"\\]|\\.)*)(";\s*)$/;
const debugLocRe = /^\s*loc\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*):(\d+):(\d+);\s*$/;
const funHeadRe = /(?:^|\s|])(?:pub\s+)?fun\s+(?:[A-Za-z_][\w.]*|"(?:[^"\\]|\\.)*")\s*\([^{]*\{\s*$/;
const instLocRe = /\bat\s+([A-Za-z_]\w*)(?:\s+in\s+[A-Za-z_]\w*)?(?:\s+holds\s+\[[^\]]*](?:\[\d+,\s*\d+])?)?;\s*$/;

type Location = {file: string; line: number; column: number};

function unescapeMirlString(raw: string): string {
    return raw.replace(/\\(\\|"|x[0-9a-fA-F]{2})/g, (_, seq: string) => {
        if (seq === '\\' || seq === '"') return seq;
        return String.fromCharCode(Number.parseInt(seq.slice(1), 16));
    });
}

function normaliseFile(raw: string): string {
    return utils.maskRootdir(raw.replaceAll('\\', '/')).replace(/^\.\//, '');
}

function normaliseDir(raw: string): string {
    const withSlash = `${raw.replaceAll('\\', '/').replace(/\/+$/, '')}/`;
    return utils.maskRootdir(withSlash).replace(/\/+$/, '') || '.';
}

/**
 * Reads MIRL modules written by `mach build --emit ir-opt` (and `--emit ir`).
 *
 * Each module carries a `debug { ... }` table of `file fN = "..."` and `loc lN = fM:line:column` records, and tags
 * instructions inside function bodies with `at lN` (before optional `in sM` and `holds [...]`). Only locations in the
 * entry editor's own file become `file: null`, which is Compiler Explorer's contract for lines owned by that editor.
 */
export class MachIrParser {
    constructor(private readonly mainFile: string) {}

    process(mirl: string): ParsedAsmResultLine[] {
        const files = new Map<string, string>();
        const locs = new Map<string, Location>();
        const result: ParsedAsmResultLine[] = [];

        let inDebug = false;
        let debugDepth = 0;
        let inFunction = false;
        let funDepth = 0;

        for (const line of utils.splitLines(mirl)) {
            if (!inDebug && !inFunction && debugOpenRe.test(line)) {
                inDebug = true;
                debugDepth = 1;
                result.push({text: line});
                continue;
            }

            if (inDebug) {
                const dirMatch = line.match(debugDirRe);
                if (dirMatch) {
                    const masked = normaliseDir(unescapeMirlString(dirMatch[2]));
                    result.push({text: `${dirMatch[1]}${masked}${dirMatch[3]}`});
                    continue;
                }

                const fileMatch = line.match(debugFileRe);
                if (fileMatch) {
                    const [, prefix, fileId, rawPath, suffix] = fileMatch;
                    const file = normaliseFile(unescapeMirlString(rawPath));
                    files.set(fileId, file);
                    const shown = file === this.mainFile ? '<source>' : file;
                    result.push({text: `${prefix}${shown}${suffix}`});
                    continue;
                }

                const locMatch = line.match(debugLocRe);
                if (locMatch) {
                    const [, locId, fileId, lineNum, colNum] = locMatch;
                    const file = files.get(fileId);
                    if (file !== undefined) {
                        locs.set(locId, {file, line: Number(lineNum), column: Number(colNum)});
                    }
                    result.push({text: line});
                    continue;
                }

                for (const ch of line) {
                    if (ch === '{') debugDepth++;
                    else if (ch === '}') debugDepth--;
                }
                if (debugDepth <= 0) inDebug = false;
                result.push({text: line});
                continue;
            }

            if (!inFunction && funHeadRe.test(line)) {
                inFunction = true;
                funDepth = 1;
                result.push({text: line});
                continue;
            }

            if (inFunction) {
                const trimmed = line.trim();
                if (trimmed === '}') {
                    funDepth--;
                    if (funDepth <= 0) inFunction = false;
                    result.push({text: line});
                    continue;
                }

                const instMatch = line.match(instLocRe);
                const loc = instMatch ? locs.get(instMatch[1]) : undefined;
                if (loc) {
                    result.push({text: line, source: this.source(loc)});
                } else {
                    result.push({text: line});
                }
                continue;
            }

            result.push({text: line});
        }

        return result;
    }

    processModules(modules: string[]): ParsedAsmResultLine[] {
        const out: ParsedAsmResultLine[] = [];
        for (const [index, mod] of modules.entries()) {
            if (index > 0 && out.length > 0 && out.at(-1)?.text !== '') {
                out.push({text: ''});
            }
            out.push(...this.process(mod));
        }
        return out;
    }

    private source(loc: Location): AsmResultSource {
        return loc.file === this.mainFile
            ? {file: null, line: loc.line, column: loc.column, mainsource: true}
            : {file: loc.file, line: loc.line, column: loc.column};
    }
}
