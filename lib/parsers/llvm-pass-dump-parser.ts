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

import _ from 'underscore';

import {
    LLVMOptPipelineBackendOptions,
    LLVMOptPipelineResults,
    Pass,
} from '../../types/compilation/llvm-opt-pipeline-output.interfaces';
import {ParseFilters} from '../../types/features/filters.interfaces';
import {ResultLine} from '../../types/resultline/resultline.interfaces';

// Note(jeremy-rifkin):
// For now this filters out a bunch of metadata we aren't interested in
// Maybe at a later date we'll want to allow user-controllable filters
// It'd be helpful to better display annotations about branch weights
// and parse debug info too at some point.

// TODO(jeremy-rifkin): Doe we already have an assert utility
function assert(condition: boolean, message?: string, ...args: any[]): asserts condition {
    if (!condition) {
        const stack = new Error('Assertion Error').stack;
        throw (
            (message
                ? `Assertion error in llvm-print-after-all-parser: ${message}`
                : `Assertion error in llvm-print-after-all-parser`) +
            (args.length > 0 ? `\n${JSON.stringify(args)}\n` : '') +
            `\n${stack}`
        );
    }
}

// Just a sanity check
function passesMatch(before: string, after: string) {
    assert(before.startsWith('IR Dump Before '));
    assert(after.startsWith('IR Dump After '));
    before = before.slice('IR Dump Before '.length);
    after = after.slice('IR Dump After '.length);
    // Observed to happen in clang 13+ for LoopDeletionPass
    if (after.endsWith(' (invalidated)')) {
        after = after.slice(0, after.length - ' (invalidated)'.length);
    }
    return before === after;
}

// Ir Dump for a pass with raw lines
type PassDump = {
    header: string;
    affectedFunction: string | undefined;
    machine: boolean;
    lines: ResultLine[];
};
// Ir Dump for a pass with raw lines broken into affected functions (or "<loop>")
type SplitPassDump = {
    header: string;
    machine: boolean;
    functions: Record<string, ResultLine[]>;
};

export class LlvmPassDumpParser {
    filters: RegExp[];
    lineFilters: RegExp[];
    debugInfoFilters: RegExp[];
    debugInfoLineFilters: RegExp[];
    irDumpHeader: RegExp;
    machineCodeDumpHeader: RegExp;
    functionDefine: RegExp;
    machineFunctionBegin: RegExp;
    functionEnd: RegExp;
    //label: RegExp;
    //instruction: RegExp;

    constructor(compilerProps) {
        //this.maxIrLines = 5000;
        //if (compilerProps) {
        //    this.maxIrLines = compilerProps('maxLinesOfAsm', this.maxIrLines);
        //}

        this.filters = [
            /^; ModuleID = '.+'$/, // module id line
            /^(source_filename|target datalayout|target triple) = ".+"$/, // module metadata
            /^; Function Attrs: .+$/, // function attributes
            /^declare .+$/, // declare directives
            /^attributes #\d+ = { .+ }$/, // attributes directive
        ];
        this.lineFilters = [
            /,? #\d+((?=( {)?$))/, // attribute annotation
        ];

        // Additional filters conditionally enabled by `filterDebugInfo`
        this.debugInfoFilters = [
            /^\s+call void @llvm.dbg.+$/, // dbg calls
            /^\s+DBG_.+$/, // dbg pseudo-instructions
            /^(!\d+) = (?:distinct )?!DI([A-Za-z]+)\(([^)]+?)\)/, // meta
            /^(!\d+) = (?:distinct )?!{.*}/, // meta
            /^(![.A-Z_a-z-]+) = (?:distinct )?!{.*}/, // meta
        ];
        this.debugInfoLineFilters = [
            /,? !dbg !\d+/, // instruction/function debug metadata
        ];

        // Ir dump headers look like "*** IR Dump After XYZ ***"
        // Machine dump headers look like "# *** IR Dump After XYZ ***:", possibly with a comment or "(function: xyz)"
        // or "(loop: %x)" at the end
        this.irDumpHeader = /^\*{3} (.+) \*{3}(?:\s+\((?:function: |loop: )(%?[\w$.]+)\))?(?:;.+)?$/;
        this.machineCodeDumpHeader = /^# \*{3} (.+) \*{3}:$/;
        // Ir dumps are "define T @_Z3fooi(...) . .. {" or "# Machine code for function _Z3fooi: <attributes>"
        // Some interesting edge cases found when testing:
        // `define internal %"struct.libassert::detail::assert_static_parameters"* @"_ZZ4mainENK3$_0clEv"(
        //      %class.anon* nonnull dereferenceable(1) %0) #5 align 2 !dbg !2 { ... }`
        // `define internal void @__cxx_global_var_init.1() #0 section ".text.startup" {`
        this.functionDefine = /^define .+ @([\w.]+|"[^"]+")\(.+$/;
        this.machineFunctionBegin = /^# Machine code for function ([\w$.]+):.*$/;
        // Functions end with either a closing brace or "# End machine code for function _Z3fooi."
        this.functionEnd = /^(?:}|# End machine code for function ([\w$.]+).)$/;
        // Either "123:" with a possible comment or something like "bb.3 (%ir-block.13):"
        //this.label = /^(?:\d+:(\s+;.+)?|\w+.+:)$/;
        //this.instruction = /^\s+.+$/;
    }

    breakdownOutputIntoPassDumps(ir: ResultLine[]) {
        // break down output by "*** IR Dump After XYZ ***" markers
        const raw_passes: PassDump[] = [];
        let pass: PassDump | null = null;
        let lastWasBlank = false; // skip duplicate blank lines
        for (const line of ir) {
            // stop once the machine code passes start, can't handle these yet
            //if (this.machineCodeDumpHeader.test(line.text)) {
            //    break;
            //}
            const irMatch = line.text.match(this.irDumpHeader);
            const machineMatch = line.text.match(this.machineCodeDumpHeader);
            const header = irMatch || machineMatch;
            if (header) {
                if (pass !== null) {
                    raw_passes.push(pass);
                }
                pass = {
                    header: header[1],
                    // in dump full module mode some headers are annotated for what function (or loop) they operate on
                    // if we aren't in full module mode or this is a header that isn't function/loop specific this will
                    // be undefined
                    affectedFunction: header[2],
                    machine: !!machineMatch,
                    lines: [],
                };
                lastWasBlank = true; // skip leading newlines after the header
            } else {
                if (pass === null) {
                    throw 'Internal error during breakdownOutput (1)';
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
            }
        }
        if (pass !== null) {
            raw_passes.push(pass);
        }
        return raw_passes;
    }

    breakdownPassDumpsIntoFunctions(dump: PassDump) {
        // break up down dumps for each pass into functions (or absence of functions in the case of loop passes)
        // we have three cases of ir dumps to consider:
        // - Most passes dump a single function
        // - Some passes dump every function
        // - Some loop passes only dump loops - these are challenging to deal with
        const pass: SplitPassDump = {
            header: dump.header,
            machine: dump.machine,
            functions: {},
        };
        let func: {
            name: string;
            lines: ResultLine[];
        } | null = null;
        for (const line of dump.lines) {
            const irFnMatch = line.text.match(this.functionDefine);
            const machineFnMatch = line.text.match(this.machineFunctionBegin);
            // function define line
            if (irFnMatch || machineFnMatch) {
                // if the last function has not been closed...
                if (func !== null) {
                    throw 'Internal error during breakdownPass (1)';
                }
                func = {
                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    name: (irFnMatch || machineFnMatch)![1],
                    lines: [line], // include the current line
                };
            } else if (line.text.startsWith('; Preheader:')) {
                // loop dump
                // every line in this dump should be part of the loop, exit condition will be end of the for loop
                assert(func === null);
                func = {
                    name: '<loop>',
                    lines: [line], // include the current line
                };
            } else {
                // close function
                if (this.functionEnd.test(line.text.trim())) {
                    // if not currently in a function
                    if (func === null) {
                        throw 'Internal error during breakdownPass (2)';
                    }
                    const {name, lines} = func;
                    lines.push(line); // include the }
                    // loop dumps can't be terminated with }
                    if (name === '<loop>') {
                        throw 'Internal error during breakdownPass (3)';
                    }
                    // somehow dumped twice?
                    if (name in pass.functions) {
                        throw 'Internal error during breakdownPass (4)';
                    }
                    pass.functions[name] = lines;
                    func = null;
                } else {
                    // lines outside a function definition
                    if (func === null) {
                        if (line.text.trim() === '') {
                            // may be a blank line
                            continue;
                        } else {
                            ///console.log('ignoring ------>', line.text);
                            // ignore
                            continue;
                        }
                    }
                    func.lines.push(line);
                }
            }
        }
        // unterminated function, either a loop dump or an error
        if (func !== null) {
            if (func.name === '<loop>') {
                // loop dumps must be alone
                if (Object.entries(pass.functions).length > 0) {
                    //console.dir(dump, { depth: 5, maxArrayLength: 100000 });
                    //console.log(pass.functions);
                    throw 'Internal error during breakdownPass (5)';
                }
                pass.functions[func.name] = func.lines;
            } else {
                throw 'Internal error during breakdownPass (6)';
            }
        }
        return pass;
    }

    breakdownIntoPassDumpsByFunction(passDumps: SplitPassDump[]) {
        // Currently we have an array of passes with a map of functions altered in each pass
        // We want to transpose to a map of functions with an array of passes on the function
        const passDumpsByFunction: Record<string, PassDump[]> = {};
        // I'm assuming loop dumps should correspond to the previous function dumped
        let previousFunction: string | null = null;
        for (const pass of passDumps) {
            const {header, machine, functions} = pass;
            const functionEntries = Object.entries(functions);
            for (const [function_name, lines] of functionEntries) {
                const name: string | null = function_name === '<loop>' ? previousFunction : function_name;
                assert(name !== null, 'Loop dump without preceding dump');
                if (!(name in passDumpsByFunction)) {
                    passDumpsByFunction[name] = [];
                }
                passDumpsByFunction[name].push({
                    header,
                    affectedFunction: undefined,
                    machine,
                    lines,
                });
            }
            if (functionEntries.length === 0) {
                // This can happen as a result of "Printing <null> Function"
                //throw 'Internal error during breakdownOutput (2)';
            } else if (functionEntries.length === 1) {
                const name = functionEntries[0][0];
                if (name !== '<loop>') {
                    previousFunction = name;
                }
            } else {
                previousFunction = null;
            }
        }
        return passDumpsByFunction;
    }

    // used for full module dump mode
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
        passDumpsByFunction['<Full Module>'] = [];
        // I'm assuming loop dumps should correspond to the previous function dumped
        //const functions = Object.keys(passDumpsByFunction);
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
                [passDumpsByFunction[fn], passDumpsByFunction['<Full Module>']].map(entry =>
                    entry.push({
                        header: `${header} (${fn})`,
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
        // We have all the passes for each function, now we will go through each function and match the before/after
        // dumps
        const final_output: LLVMOptPipelineResults = {};
        for (const [function_name, passDumps] of Object.entries(passDumpsByFunction)) {
            // I had a fantastic chunk2 method to iterate the passes in chunks of 2 but I've been foiled by an edge
            // case: At least the "Delete dead loops" may only print a before dump and no after dump
            const passes: Pass[] = [];
            // i incremented appropriately later
            for (let i = 0; i < passDumps.length; ) {
                const pass: Pass = {
                    name: '',
                    machine: false,
                    after: [],
                    before: [],
                    irChanged: true,
                };
                const current_dump = passDumps[i];
                const next_dump = i < passDumps.length - 1 ? passDumps[i + 1] : null;
                if (current_dump.header.startsWith('IR Dump After ')) {
                    // An after dump without a before dump - I don't think this can happen but we'll handle just in case
                    pass.name = current_dump.header.slice('IR Dump After '.length);
                    pass.after = current_dump.lines;
                    i++;
                } else if (current_dump.header.startsWith('IR Dump Before ')) {
                    if (next_dump !== null && next_dump.header.startsWith('IR Dump After ')) {
                        assert(
                            passesMatch(current_dump.header, next_dump.header),
                            '',
                            current_dump.header,
                            next_dump.header,
                        );
                        assert(current_dump.machine === next_dump.machine);
                        pass.name = current_dump.header.slice('IR Dump Before '.length);
                        pass.before = current_dump.lines;
                        pass.after = next_dump.lines;
                        i += 2;
                    } else {
                        // Before with no after - this can happen with Delete dead loops
                        pass.name = current_dump.header.slice('IR Dump Before '.length);
                        pass.before = current_dump.lines;
                        i++;
                    }
                } else {
                    assert(false, 'Unexpected pass header', current_dump.header);
                }
                pass.machine = current_dump.machine;

                // The first machine pass outputs the same MIR before and after,
                // making it seem like it did nothing.
                // Assuming we ran some IR pass before this, grab its output as
                // the before text, ensuring the first MIR pass appears when
                // inconsequential passes are filtered away.
                const previousPass = passes.at(-1);
                if (previousPass && previousPass.machine !== pass.machine) {
                    pass.before = previousPass.after;
                }

                // check for equality
                pass.irChanged = pass.before.map(x => x.text).join('\n') !== pass.after.map(x => x.text).join('\n');
                passes.push(pass);
            }
            //console.dir(passes, {
            //    depth: 5,
            //    maxArrayLength: 100000
            //});
            assert(!(function_name in final_output), 'xxx');
            final_output[function_name] = passes;
        }
        return final_output;
    }

    breakdownOutput(ir: ResultLine[], llvmOptPipelineOptions: LLVMOptPipelineBackendOptions) {
        // break down output by "*** IR Dump After XYZ ***" markers
        const raw_passes = this.breakdownOutputIntoPassDumps(ir);
        if (llvmOptPipelineOptions.fullModule) {
            const passDumpsByFunction = this.associateFullDumpsWithFunctions(raw_passes);
            // Match before / after pass dumps and we're done
            return this.matchPassDumps(passDumpsByFunction);
        } else {
            // Further break down by functions in each dump
            const passDumps = raw_passes.map(this.breakdownPassDumpsIntoFunctions.bind(this));
            // Transform array of passes containing multiple functions into a map from functions to arrays of passes on
            // those functions
            const passDumpsByFunction = this.breakdownIntoPassDumpsByFunction(passDumps);
            // Match before / after pass dumps and we're done
            return this.matchPassDumps(passDumpsByFunction);
        }
    }

    // findLastIndex is currently unavailable in node, replacement essentially taken from
    // https://stackoverflow.com/a/53187807/89706
    findLastIndex<T>(array: Array<T>, predicate: (value: T) => boolean): number {
        let l = array.length;
        while (l--) {
            if (predicate(array[l])) return l;
        }
        return 0; // if not found slice entire array. -1 would slice only the *last* element
    }

    applyIrFilters(ir: ResultLine[], llvmOptPipelineOptions: LLVMOptPipelineBackendOptions) {
        // Additional filters conditionally enabled by `filterDebugInfo`
        let filters = this.filters;
        let lineFilters = this.lineFilters;
        if (llvmOptPipelineOptions.filterDebugInfo) {
            filters = filters.concat(this.debugInfoFilters);
            lineFilters = lineFilters.concat(this.debugInfoLineFilters);
        }

        // Filter junk
        // prettier-ignore
        const idxAfterHeaders = this.findLastIndex(
            ir,
            line => (line.text.match(this.irDumpHeader) || line.text.match(this.machineCodeDumpHeader)) !== null);

        return ir
            .slice(idxAfterHeaders)
            .filter(line => filters.every(re => line.text.match(re) === null)) // apply filters
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
            });
    }

    process(ir: ResultLine[], _: ParseFilters, llvmOptPipelineOptions: LLVMOptPipelineBackendOptions) {
        const preprocessed_lines = this.applyIrFilters(ir, llvmOptPipelineOptions);
        return this.breakdownOutput(preprocessed_lines, llvmOptPipelineOptions);
    }
}
