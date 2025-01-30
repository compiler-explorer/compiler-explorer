// Copyright (c) 2023, Compiler Explorer Authors
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
import {assert} from '../assert.js';
import {PropertyGetter} from '../properties.interfaces.js';

type PassDump = {
    header: string;
    group: string | undefined;
    machine: boolean;
    lines: ResultLine[];
};

export class RacketPassDumpParser {
    filters: RegExp[];
    lineFilters: RegExp[];
    debugInfoFilters: RegExp[];
    debugInfoLineFilters: RegExp[];
    metadataLineFilters: RegExp[];
    moduleHeader: RegExp;
    linkletPhaseHeader: RegExp;
    linkletHeader: RegExp;
    stepHeader: RegExp;
    passHeader: RegExp;
    mainModule: RegExp;

    constructor(compilerProps: PropertyGetter) {
        // Filters that are always enabled
        this.filters = [];
        this.lineFilters = [];

        // Additional filters conditionally enabled by `filterDebugInfo`
        this.debugInfoFilters = [];
        this.debugInfoLineFilters = [];

        // Conditionally enabled by `filterIRMetadata`
        this.metadataLineFilters = [];

        // Racket's compilation pipeline (and thus its pass output)
        // works on one module at a time.
        this.moduleHeader = /^[ ;]*?compile-linklet: module: (.+)$/;

        // For each module Racket compiles, several linklets are produced that
        // cover different aspects of the module. See Racket's
        // `compile/module.rkt` for more details. The various kinds of linklets
        // for each module include:
        //   - Body linklets for each phase (keyed by phase number)
        //   - `decl`: declaration linklet (shared across module instances)
        //   - `data`: data linklet (shared across module instances)
        //   - `stx`: syntax literals linklet
        //   - `stx-data`: syntax literals data linklet (shared across module
        //     instances)
        this.linkletHeader = /^[ ;]*?compile-linklet: name: (.+)$/;
        this.linkletPhaseHeader = /^[ ;]*?compile-linklet: phase: (\d+)$/;

        // Each linklet moves through various compilation steps.
        this.stepHeader = /^[ ;]*?compile-linklet: step: (.+)$/;
        this.passHeader = /^[ ;]*?output of (.+):$/;

        // Modules with `main` at the top-level are actually showing the
        // compilation process (bits of `raco make`), so we filter those out.
        this.mainModule = /^\(?main/;
    }

    breakdownOutputIntoPassDumps(logLines: ResultLine[]) {
        // Collect output from each compilation step
        const rawPasses: PassDump[] = [];
        let pass: PassDump | null = null;

        // Progressively assemble a group name by cobbling together the
        // module and linklet names.
        let mod: string | null = null;
        let linklet: string | null = null;
        let linkletPhase: number | null = null;

        let lastWasBlank = false; // skip duplicate blank lines

        for (const line of logLines) {
            const moduleMatch = line.text.match(this.moduleHeader);
            if (moduleMatch) {
                mod = moduleMatch[1];
                continue;
            }
            const linkletMatch = line.text.match(this.linkletHeader);
            if (linkletMatch) {
                linklet = linkletMatch[1];
                continue;
            }
            const linkletPhaseMatch = line.text.match(this.linkletPhaseHeader);
            if (linkletPhaseMatch) {
                linkletPhase = parseInt(linkletPhaseMatch[1]);
                continue;
            }
            const stepMatch = line.text.match(this.stepHeader);
            const passMatch = line.text.match(this.passHeader);
            if (stepMatch || passMatch) {
                const name = stepMatch?.[1] || passMatch?.[1];
                assert(name);
                if (pass !== null) {
                    rawPasses.push(pass);
                    pass = null;
                }
                if (mod?.match(this.mainModule)) {
                    // Skip modules with `main` at the root
                    continue;
                }
                let moduleAndLinkletName = `module: ${mod}, linklet: ${linklet}`;
                if (linklet === 'module' && linkletPhase !== null) {
                    moduleAndLinkletName += `, phase: ${linkletPhase}`;
                }
                pass = {
                    header: name,
                    group: moduleAndLinkletName,
                    machine: false,
                    lines: [],
                };
                lastWasBlank = true; // skip leading newlines after the header
            } else {
                if (!pass) {
                    continue;
                }
                if (line.text.startsWith(';;')) {
                    // Skip any other header lines
                    continue;
                }
                if (line.text.trim() === '') {
                    if (!lastWasBlank) {
                        pass.lines.push(line);
                    }
                    lastWasBlank = true;
                } else {
                    pass.lines.push(line);
                    lastWasBlank = false;
                }
                if (line.text === 'done') {
                    // The last step just emits "done", so stop once we've seen
                    // it. This conveniently drops any trailing logs we don't
                    // want as well.
                    rawPasses.push(pass);
                    pass = null;
                }
            }
        }
        if (pass !== null) {
            rawPasses.push(pass);
        }
        return rawPasses;
    }

    associatePassDumpsWithGroups(passDumps: PassDump[]) {
        const passDumpsByGroup: Record<string, PassDump[]> = {};
        // First figure out what all the groups are
        for (const pass of passDumps) {
            if (pass.group) {
                passDumpsByGroup[pass.group] = [];
            }
        }
        for (const pass of passDumps) {
            const {header, group, machine, lines} = pass;
            if (group) {
                assert(group in passDumpsByGroup);
                [passDumpsByGroup[group]].map(entry =>
                    entry.push({
                        header,
                        group,
                        machine,
                        lines,
                    }),
                );
            } else {
                // applies to everything
                for (const [_, entry] of Object.entries(passDumpsByGroup)) {
                    entry.push({
                        header,
                        group: undefined,
                        machine,
                        lines,
                    });
                }
            }
        }
        return passDumpsByGroup;
    }

    matchPassDumps(passDumpsByGroup: Record<string, PassDump[]>) {
        // We have collected output for each step
        // grouped by module and linklet name
        // We now assemble them into before / after pairs
        const finalOutput: OptPipelineResults = {};
        for (const [group, passDumps] of Object.entries(passDumpsByGroup)) {
            const passes: Pass[] = [];
            for (let i = 0; i < passDumps.length; i++) {
                const pass: Pass = {
                    name: '',
                    machine: false,
                    before: [],
                    after: [],
                    irChanged: true,
                };
                const previousDump = i > 0 ? passDumps[i - 1] : null;
                const currentDump = passDumps[i];

                pass.name = currentDump.header;
                if (previousDump) {
                    pass.before = previousDump.lines;
                }
                pass.after = currentDump.lines;

                // check for equality
                pass.irChanged = pass.before.map(x => x.text).join('\n') !== pass.after.map(x => x.text).join('\n');
                passes.push(pass);
            }
            finalOutput[group] = passes;
        }
        return finalOutput;
    }

    breakdownOutput(ir: ResultLine[], llvmOptPipelineOptions: OptPipelineBackendOptions) {
        const rawPasses = this.breakdownOutputIntoPassDumps(ir);
        const passDumpsByGroup = this.associatePassDumpsWithGroups(rawPasses);
        // Match before / after pass dumps and we're done
        return this.matchPassDumps(passDumpsByGroup);
    }

    applyIrFilters(ir: ResultLine[], optPipelineOptions: OptPipelineBackendOptions) {
        // Additional filters conditionally enabled by `filterDebugInfo`/`filterIRMetadata`
        let filters = this.filters;
        let lineFilters = this.lineFilters;
        if (optPipelineOptions.filterDebugInfo) {
            filters = filters.concat(this.debugInfoFilters);
            lineFilters = lineFilters.concat(this.debugInfoLineFilters);
        }
        if (optPipelineOptions.filterIRMetadata) {
            lineFilters = lineFilters.concat(this.metadataLineFilters);
        }

        return (
            ir
                // whole-line filters
                .filter(line => filters.every(re => line.text.match(re) === null))
                // intra-line filters
                .map(resultLine => {
                    let line = resultLine.text;
                    // eslint-disable-next-line no-constant-condition
                    while (true) {
                        let newLine = line;
                        for (const re of lineFilters) {
                            newLine = newLine.replace(re, '');
                        }
                        if (newLine === line) {
                            break;
                        } else {
                            line = newLine;
                        }
                    }
                    resultLine.text = line;
                    return resultLine;
                })
        );
    }

    process(output: ResultLine[], _: ParseFiltersAndOutputOptions, optPipelineOptions: OptPipelineBackendOptions) {
        const preprocessedLines = this.applyIrFilters(output, optPipelineOptions);
        return this.breakdownOutput(preprocessedLines, optPipelineOptions);
    }
}
