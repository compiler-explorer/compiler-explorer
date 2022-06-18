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

import {LLVMOptPipelineOutput, OutputLine, Pass} from '../../types/compilation/llvm-opt-pipeline-output.interfaces';
import {ParseFilters} from '../../types/features/filters.interfaces';
import * as utils from '../utils';

// Note(jeremy-rifkin):
// For now this filters out a bunch of metadata we aren't interested in
// Maybe at a later date we'll want to allow user-controllable filters
// It'd be helpful to better display annotations about branch weights
// and parse debug info too at some point.

// TODO(jeremy-rifkin): Doe we already have an assert utility
function assert(condition: boolean, message?: string, ...args: any[]): asserts condition {
    if (!condition) {
        const error = message
            ? `Assertion error in llvm-print-after-all-parser: ${message}`
            : `Assertion error in llvm-print-after-all-parser`;
        console.log(error, ...args);
        console.trace();
        throw error;
    }
}

// Ir Dump for a pass with raw lines
type PassDump = {
    header: string;
    machine: boolean;
    lines: OutputLine[];
};
// Ir Dump for a pass with raw lines broken into affected functions (or "<loop>")
type SplitPassDump = {
    header: string;
    machine: boolean;
    functions: Record<string, OutputLine[]>;
};

export class LlvmPassDumpParser {
    filters: RegExp[];
    lineFilters: RegExp[];
    irDumpAfterHeader: RegExp;
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
            /^\s+call void @llvm.dbg.value.+$/, // dbg calls
            /^\s+call void @llvm.dbg.declare.+$/, // dbg calls
            /^declare .+$/, // declare directives
            /^(!\d+) = (?:distinct )?!DI([A-Za-z]+)\(([^)]+?)\)/, // meta
            /^(!\d+) = (?:distinct )?!{.*}/, // meta
            /^(![.A-Z_a-z-]+) = (?:distinct )?!{.*}/, // meta
            /^attributes #\d+ = { .+ }$/, // attributes directive
        ];
        this.lineFilters = [
            /,? ![\dA-Za-z]+((?=( {)?$))/, // debug annotation
            /,? #\d+((?=( {)?$))/, // attribute annotation
        ];

        // Ir dump headers look like "*** IR Dump After XYZ ***"
        // Machine dump headers look like "# *** IR Dump After XYZ ***:", possibly with a comment or "(function: xyz)"
        // or "(loop: %x)" at the end
        this.irDumpAfterHeader = /^\*{3} (.+) \*{3}(\s+\((function: \w+|loop: %\w+)\))?(;.+)?$/;
        this.machineCodeDumpHeader = /^# \*{3} (.+) \*{3}:$/;
        // Ir dumps are "define T @_Z3fooi(...) . .. {" or "# Machine code for function _Z3fooi: <attributes>"
        this.functionDefine = /^define .+ @(\w+)\(.+$/;
        this.machineFunctionBegin = /^# Machine code for function (\w+):.*$/;
        // Functions end with either a closing brace or "# End machine code for function _Z3fooi."
        this.functionEnd = /^(?:}|# End machine code for function (\w+).)$/;
        // Either "123:" with a possible comment or something like "bb.3 (%ir-block.13):"
        //this.label = /^(?:\d+:(\s+;.+)?|\w+.+:)$/;
        //this.instruction = /^\s+.+$/;
    }

    breakdownOutputIntoPassDumps(ir: OutputLine[]) {
        // break down output by "*** IR Dump After XYZ ***" markers
        const raw_passes: PassDump[] = [];
        let pass: PassDump | null = null;
        for (const line of ir) {
            // stop once the machine code passes start, can't handle these yet
            //if (this.machineCodeDumpHeader.test(line.text)) {
            //    break;
            //}
            const irMatch = line.text.match(this.irDumpAfterHeader);
            const machineMatch = line.text.match(this.machineCodeDumpHeader);
            if (irMatch || machineMatch) {
                if (pass !== null) {
                    raw_passes.push(pass);
                }
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                pass = {
                    header: (irMatch || machineMatch)![1],
                    machine: !!machineMatch,
                    lines: [],
                };
            } else {
                if (pass === null) {
                    throw 'Internal error during breakdownOutput (1)';
                }
                pass.lines.push(line);
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
            lines: OutputLine[];
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
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                func = {
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
                            console.log('ignoring ------>', line.text);
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

    matchPassDumps(passDumpsByFunction: Record<string, PassDump[]>) {
        // We have all the passes for each function, now we will go through each function and match the before/after
        // dumps
        const final_output: LLVMOptPipelineOutput = {};
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
                assert(i < passDumps.length - 1); // make sure there's another item
                const current_dump = passDumps[i];
                const next_dump = passDumps[i + 1];
                if (current_dump.header.startsWith('IR Dump After ')) {
                    // An after dump without a before dump - I don't think this can happen but we'll handle just in case
                    pass.name = current_dump.header.slice('IR Dump After '.length);
                    pass.after = current_dump.lines;
                    i++;
                } else if (current_dump.header.startsWith('IR Dump Before ')) {
                    if (next_dump.header.startsWith('IR Dump After ')) {
                        assert(
                            current_dump.header.slice('IR Dump Before '.length) ===
                                next_dump.header.slice('IR Dump After '.length),
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

    breakdownOutput(ir: OutputLine[]) {
        // break down output by "*** IR Dump After XYZ ***" markers
        const raw_passes = this.breakdownOutputIntoPassDumps(ir);
        // Further break down by functions in each dump
        const passDumps = raw_passes.map(this.breakdownPassDumpsIntoFunctions.bind(this));
        // Transform array of passes containing multiple functions into a map from functions to arrays of passes on
        // those functions
        const passDumpsByFunction = this.breakdownIntoPassDumpsByFunction(passDumps);
        // Match before / after pass dumps and we're done
        return this.matchPassDumps(passDumpsByFunction);
    }

    process(ir: OutputLine[], _: ParseFilters) {
        // Filter a lot of junk before processing
        const preprocessed_lines = ir
            .filter(line => this.filters.every(re => line.text.match(re) === null)) // apply filters
            .map(_line => {
                let line = _line.text;
                // eslint-disable-next-line no-constant-condition
                while (true) {
                    let newLine = line;
                    for (const re of this.lineFilters) {
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

        return this.breakdownOutput(preprocessed_lines);
    }
}
