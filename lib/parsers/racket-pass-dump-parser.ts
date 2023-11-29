// Copyright (c) 2022, Compiler Explorer Authors
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

type PassDump = {
    header: string;
    affectedFunction: string | undefined;
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

    constructor(compilerProps) {
        // Filters that are always enabled
        this.filters = [
        ];
        this.lineFilters = [
        ];

        // Additional filters conditionally enabled by `filterDebugInfo`
        this.debugInfoFilters = [
        ];
        this.debugInfoLineFilters = [
        ];

        // Conditionally enabled by `filterIRMetadata`
        this.metadataLineFilters = [
        ];

        // Racket's compilation pipeline (and thus its pass output)
        // works on one module at a time.
        this.moduleHeader = /^[; ]*?compile-linklet: module: (.+)$/;

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
        this.linkletHeader = /^[; ]*?compile-linklet: name: (.+)$/;
        this.linkletPhaseHeader = /^[; ]*?compile-linklet: phase: (\d+)$/;

        // Each linklet moves through various compilation steps.
        this.stepHeader = /^[; ]*?compile-linklet: step: (.+)$/;
        this.passHeader = /^[; ]*?output of (.+):$/;

        // Modules with `main` at the top-level are actually showing the
        // compilation process (bits of `raco make`), so we filter those out.
        this.mainModule = /^\(?main/;
    }

    breakdownOutputIntoPassDumps(logLines: ResultLine[]) {
        // Collect output from each compilation step
        const raw_passes: PassDump[] = [];
        let pass: PassDump | null = null;

        // Progressively assemble a faux-function name by cobbling together the
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
                    raw_passes.push(pass);
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
                    affectedFunction: moduleAndLinkletName,
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
                    raw_passes.push(pass);
                    pass = null;
                }
            }
        }
        if (pass !== null) {
            raw_passes.push(pass);
        }
        return raw_passes;
    }

    associateFullDumpsWithFunctions(passDumps: PassDump[]) {
        // Currently we have an array of passes that'll have target annotations
        const passDumpsByFunction: Record<string, PassDump[]> = {};
        // First figure out what all the functions are
        for (const pass of passDumps) {
            // If it starts with a % it's a loop
            if (pass.affectedFunction && !pass.affectedFunction.startsWith('%')) {
                passDumpsByFunction[pass.affectedFunction] = [];
            }
        }
        let previousFunction: string | null = null;
        for (const pass of passDumps) {
            const {header, affectedFunction, machine, lines} = pass;
            if (affectedFunction) {
                let fn = affectedFunction;
                if (affectedFunction.startsWith('%')) {
                    assert(previousFunction !== null);
                    fn = previousFunction;
                }
                assert(fn in passDumpsByFunction);
                [passDumpsByFunction[fn]].map(entry =>
                    entry.push({
                        header,
                        affectedFunction: fn,
                        machine,
                        lines,
                    }),
                );
                previousFunction = fn;
            } else {
                // applies to everything
                for (const [_, entry] of Object.entries(passDumpsByFunction)) {
                    entry.push({
                        header,
                        affectedFunction: undefined,
                        machine,
                        lines,
                    });
                }
                previousFunction = null;
            }
        }
        return passDumpsByFunction;
    }

    matchPassDumps(passDumpsByFunction: Record<string, PassDump[]>) {
        // We have collected output for each step
        // grouped by "function" (module and linklet name)
        // We now assemble them into before / after pairs
        const final_output: OptPipelineResults = {};
        for (const [function_name, passDumps] of Object.entries(passDumpsByFunction)) {
            const passes: Pass[] = [];
            for (let i = 0; i < passDumps.length; i++) {
                const pass: Pass = {
                    name: '',
                    machine: false,
                    before: [],
                    after: [],
                    irChanged: true,
                };
                const previous_dump = i > 0 ? passDumps[i - 1] : null;
                const current_dump = passDumps[i];

                pass.name = current_dump.header;
                if (previous_dump) {
                    pass.before = previous_dump.lines;
                }
                pass.after = current_dump.lines;

                // check for equality
                pass.irChanged = pass.before.map(x => x.text).join('\n') !== pass.after.map(x => x.text).join('\n');
                passes.push(pass);
            }
            // console.dir(passes, {
            //    depth: 5,
            //    maxArrayLength: 100000
            // });
            final_output[function_name] = passes;
        }
        return final_output;
    }

    breakdownOutput(ir: ResultLine[], llvmOptPipelineOptions: OptPipelineBackendOptions) {
        const raw_passes = this.breakdownOutputIntoPassDumps(ir);
        const passDumpsByFunction = this.associateFullDumpsWithFunctions(raw_passes);
        // Match before / after pass dumps and we're done
        return this.matchPassDumps(passDumpsByFunction);
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
                .map(_line => {
                    let line = _line.text;
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
                    _line.text = line;
                    return _line;
                })
        );
    }

    process(
        output: ResultLine[],
        _: ParseFiltersAndOutputOptions,
        optPipelineOptions: OptPipelineBackendOptions,
    ) {
        const preprocessed_lines = this.applyIrFilters(output, optPipelineOptions);
        return this.breakdownOutput(preprocessed_lines, optPipelineOptions);
    }
}
