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

import {
    OptPipelineBackendOptions,
    OptPipelineResults,
    Pass,
} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {assert} from '../assert.js';
import {PropertyGetter} from '../properties.interfaces.js';

// Helper function to extract pass name from header
function extractPassName(header: string): string {
    if (header.startsWith('IR Dump Before ')) {
        return header.slice('IR Dump Before '.length);
    }

    if (header.startsWith('IR Dump After ')) {
        let passName = header.slice('IR Dump After '.length);
        // Handle invalidated passes
        if (passName.endsWith(' (invalidated)')) {
            passName = passName.slice(0, passName.length - ' (invalidated)'.length);
        }
        return passName;
    }
    assert(false, 'Unexpected pass header', header);
}

// Ir Dump for a pass with raw lines
type PassDump = {
    header: string;
    affectedFunction: string | undefined;
    lines: ResultLine[];
};

// Ir Dump for a pass with raw lines broken into affected functions
type SplitPassDump = {
    header: string;
    functions: Record<string, ResultLine[]>;
};

export class MlirPassDumpParser {
    locationDefine: RegExp;
    locationReference: RegExp;
    irDumpHeader: RegExp;
    functionDefine: RegExp;
    moduleDefine: RegExp;
    functionEnd: RegExp;
    moduleEnd: RegExp;

    constructor(compilerProps: PropertyGetter) {
        // Location definitions: #loc0 = loc(...)
        this.locationDefine = /^#loc\d* = loc\(.+\)$/;

        // Location references: loc(#loc0), loc("/app/example.py":19:0), loc(unknown)
        this.locationReference = /\s*loc\([^)]*\)/g;

        // MLIR dump headers look like "// -----// IR Dump Before/After XYZ (xyz) ('operation-type' operation: @function_name) //----- //"
        this.irDumpHeader = /^\/\/ -----\/\/ (IR Dump (?:Before|After) .+) \/\/----- \/\/$/;

        // MLIR function definitions look like "func.func @function_name(...) {"
        // or "tt.func public @function_name(...) {"
        this.functionDefine = /^\s*(\w+\.func\s+(?:\w+\s+)?@(\w+).*\{)$/;

        // MLIR module definitions look like "module {"
        this.moduleDefine = /^\s*(module\s*\{)$/;

        // Functions end with a closing brace
        this.functionEnd = /^\s*\}\s*$/;

        // Modules end with a closing brace
        this.moduleEnd = /^\s*\}\s*$/;
    }

    breakdownOutputIntoPassDumps(ir: ResultLine[]) {
        // break down output by "// -----// IR Dump Before/After XYZ //----- //" markers
        const raw_passes: PassDump[] = [];
        let pass: PassDump | null = null;
        let lastWasBlank = false; // skip duplicate blank lines

        for (const line of ir) {
            const irMatch = line.text.match(this.irDumpHeader);

            if (irMatch) {
                if (pass !== null) {
                    raw_passes.push(pass);
                }

                const headerText = irMatch[1];
                pass = {
                    header: headerText,
                    affectedFunction: undefined,
                    lines: [],
                };
                lastWasBlank = true; // skip leading newlines after the header
            } else {
                if (pass === null) continue;

                if (line.text.trim() === '') {
                    if (!lastWasBlank) {
                        pass.lines.push(line);
                    }
                    lastWasBlank = true;
                } else {
                    pass.lines.push(line);
                    lastWasBlank = false;
                }
            }
        }

        if (pass !== null) {
            raw_passes.push(pass);
        }

        return raw_passes;
    }

    breakdownPassDumpsIntoFunctions(dump: PassDump) {
        // Simplified version based on the assumption that:
        // 1. Functions always live inside a single module
        // 2. A single module always has a single function in it
        // 3. We use the name of the function to show the entire module
        const pass: SplitPassDump = {
            header: dump.header,
            functions: {},
        };

        // Find the function name inside the module
        let functionName: string | null = null;
        for (const line of dump.lines) {
            const funcMatch = line.text.match(this.functionDefine);
            if (funcMatch) {
                functionName = funcMatch[2];
                break;
            }
        }

        // If we found a function name, use it; otherwise use "module"
        const name = functionName || 'module';
        pass.functions[name] = dump.lines;

        return pass;
    }

    breakdownIntoPassDumpsByFunction(passDumps: SplitPassDump[]) {
        // Currently we have an array of passes with a map of functions altered in each pass
        // We want to transpose to a map of functions with an array of passes on the function
        const passDumpsByFunction: Record<string, PassDump[]> = {};

        for (const pass of passDumps) {
            const {header, functions} = pass;

            for (const [function_name, lines] of Object.entries(functions)) {
                if (!(function_name in passDumpsByFunction)) {
                    passDumpsByFunction[function_name] = [];
                }

                passDumpsByFunction[function_name].push({
                    header,
                    affectedFunction: undefined,
                    lines,
                });
            }
        }

        return passDumpsByFunction;
    }

    associateFullDumpsWithFunctions(passDumps: PassDump[]) {
        // Currently we have an array of passes that'll have target annotations
        const passDumpsByFunction: Record<string, PassDump[]> = {};

        // First figure out what all the functions are
        for (const pass of passDumps) {
            if (pass.affectedFunction) {
                passDumpsByFunction[pass.affectedFunction] = [];
            }
        }

        // Add a special entry for the full module
        passDumpsByFunction['<Full Module>'] = [];

        for (const pass of passDumps) {
            const {header, affectedFunction, lines} = pass;

            if (affectedFunction) {
                // This pass affects a specific function
                assert(affectedFunction in passDumpsByFunction);

                // Add to both the specific function and the full module view
                [passDumpsByFunction[affectedFunction], passDumpsByFunction['<Full Module>']].map(entry =>
                    entry.push({
                        header: `${header} (${affectedFunction})`,
                        affectedFunction,
                        lines,
                    }),
                );
            } else {
                // This pass applies to everything
                for (const [_, entry] of Object.entries(passDumpsByFunction)) {
                    entry.push({
                        header,
                        affectedFunction: undefined,
                        lines,
                    });
                }
            }
        }

        return passDumpsByFunction;
    }

    isIrChanged(before: ResultLine[], after: ResultLine[]) {
        if (before.length !== after.length) {
            return true;
        }
        for (let i = 0; i < before.length; i++) {
            if (before[i].text !== after[i].text) {
                return true;
            }
        }
        return false;
    }

    matchPassDumps(passDumpsByFunction: Record<string, PassDump[]>) {
        // We have all the passes for each function, now we will go through each function and match the before/after
        // dumps, handling the case where the same pass might occur multiple times
        const final_output: OptPipelineResults = {};

        for (const [function_name, passDumps] of Object.entries(passDumpsByFunction)) {
            const passes: Pass[] = [];

            // Use a stack of "Before" passes
            const beforePasses: PassDump[] = [];

            // Collect all "Before" passes in order
            for (const dump of passDumps) {
                if (dump.header.startsWith('IR Dump Before ')) {
                    beforePasses.push(dump);
                }
            }

            // Process "After" passes and match with "Before" passes
            for (const dump of passDumps) {
                if (dump.header.startsWith('IR Dump After ')) {
                    const afterPassName = extractPassName(dump.header);
                    const pass: Pass = {
                        name: afterPassName,
                        machine: false,
                        after: dump.lines,
                        before: [],
                        irChanged: true,
                    };

                    // Find matching "Before" pass by name
                    for (let i = 0; i < beforePasses.length; i++) {
                        const beforePassName = extractPassName(beforePasses[i].header);
                        if (beforePassName === afterPassName) {
                            // Found a match, use it and remove from the stack
                            pass.before = beforePasses[i].lines;

                            // Check for equality
                            pass.irChanged = this.isIrChanged(pass.before, pass.after);

                            // Remove the matched "Before" pass
                            beforePasses.splice(i, 1);
                            break;
                        }
                    }

                    passes.push(pass);
                }
            }

            // If we only have before passes (no after passes), diff between consecutive before passes
            // This happened in Triton since it sets enableIRPrinting(printAfterOnlyOnFailure=false)
            if (passes.length === 0) {
                for (let i = 0; i < beforePasses.length - 1; i++) {
                    const isLast = i === beforePasses.length - 1;
                    const passName = extractPassName(beforePasses[i].header);
                    const before = beforePasses[i].lines;
                    const after = isLast ? beforePasses[i].lines : beforePasses[i + 1].lines;
                    const irChanged = isLast ? false : this.isIrChanged(before, after);
                    const pass: Pass = {
                        name: passName,
                        machine: false,
                        before: before,
                        after: after,
                        irChanged: irChanged,
                    };
                    passes.push(pass);
                }
            } else {
                // Handle any remaining "Before" passes that don't have corresponding "After" passes
                for (const beforeDump of beforePasses) {
                    const passName = extractPassName(beforeDump.header);
                    const pass: Pass = {
                        name: passName,
                        machine: false,
                        before: beforeDump.lines,
                        after: [],
                        irChanged: true, // Assume changed since there's no "After" to compare with
                    };
                    passes.push(pass);
                }
            }

            final_output[function_name] = passes;
        }

        return final_output;
    }

    breakdownOutput(ir: ResultLine[], optPipelineOptions: OptPipelineBackendOptions) {
        // break down output by "// -----// IR Dump Before/After XYZ //----- //" markers
        const raw_passes = this.breakdownOutputIntoPassDumps(ir);

        if (optPipelineOptions.fullModule) {
            const passDumpsByFunction = this.associateFullDumpsWithFunctions(raw_passes);
            // Match before / after pass dumps and we're done
            return this.matchPassDumps(passDumpsByFunction);
        }

        // Further break down by functions in each dump
        const passDumps = raw_passes.map(this.breakdownPassDumpsIntoFunctions.bind(this));

        // Transform array of passes containing multiple functions into a map from functions to arrays of passes on
        // those functions
        const passDumpsByFunction = this.breakdownIntoPassDumpsByFunction(passDumps);

        // Match before / after pass dumps and we're done
        return this.matchPassDumps(passDumpsByFunction);
    }

    applyIrFilters(ir: ResultLine[]) {
        return ir
            .filter(line => line.text.match(this.locationDefine) === null)
            .map(line => ({
                ...line,
                text: line.text.replace(this.locationReference, ''),
            }));
    }

    process(output: ResultLine[], _: ParseFiltersAndOutputOptions, optPipelineOptions: OptPipelineBackendOptions) {
        // Crop out any junk before the pass dumps
        const ir = output.slice(output.findIndex(line => this.irDumpHeader.test(line.text)));

        const preprocessed_lines = this.applyIrFilters(ir);
        return this.breakdownOutput(preprocessed_lines, optPipelineOptions);
    }
}
