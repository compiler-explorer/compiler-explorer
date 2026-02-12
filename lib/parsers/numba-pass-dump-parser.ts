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

import type {
    OptPipelineBackendOptions,
    OptPipelineResults,
    Pass,
} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';

// Numba's NUMBA_DEBUG_PRINT_AFTER=all produces headers like:
//   ------__main__.example: nopython: AFTER translate_bytecode------
// Variable-width dashes pad each side to a fixed total width.
const passHeader = /^-{2,}\s*(.+?):\s*(nopython|object):\s*AFTER\s+(.+?)\s*-{2,}$/;

type PassDump = {
    passName: string;
    functionName: string;
    lines: ResultLine[];
};

export class NumbaPassDumpParser {
    breakdownOutputIntoPassDumps(logLines: ResultLine[]): PassDump[] {
        const dumps: PassDump[] = [];
        let current: PassDump | null = null;

        for (const line of logLines) {
            const match = line.text.match(passHeader);
            if (match) {
                if (current) {
                    dumps.push(current);
                }
                current = {
                    functionName: match[1],
                    passName: match[3],
                    lines: [],
                };
                continue;
            }
            if (current) {
                current.lines.push(line);
            }
        }
        if (current) {
            dumps.push(current);
        }
        return dumps;
    }

    associatePassDumpsWithGroups(dumps: PassDump[]): Record<string, PassDump[]> {
        const grouped: Record<string, PassDump[]> = {};
        for (const dump of dumps) {
            if (!(dump.functionName in grouped)) {
                grouped[dump.functionName] = [];
            }
            grouped[dump.functionName].push(dump);
        }
        return grouped;
    }

    matchPassDumps(grouped: Record<string, PassDump[]>): OptPipelineResults {
        const results: OptPipelineResults = {};
        for (const [group, dumps] of Object.entries(grouped)) {
            const passes: Pass[] = [];
            for (let i = 0; i < dumps.length; i++) {
                const prev = i > 0 ? dumps[i - 1] : null;
                const curr = dumps[i];
                const before = prev ? prev.lines : [];
                const after = curr.lines;
                const irChanged = before.map(l => l.text).join('\n') !== after.map(l => l.text).join('\n');
                passes.push({
                    name: curr.passName,
                    machine: false,
                    before,
                    after,
                    irChanged,
                });
            }
            results[group] = passes;
        }
        return results;
    }

    process(
        output: ResultLine[],
        _filters: ParseFiltersAndOutputOptions,
        _optPipelineOptions: OptPipelineBackendOptions,
    ): OptPipelineResults {
        const dumps = this.breakdownOutputIntoPassDumps(output);
        const grouped = this.associatePassDumpsWithGroups(dumps);
        return this.matchPassDumps(grouped);
    }
}
