// Copyright (c) 2018, Compiler Explorer Authors
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

type OutputLine = {text: string};
// Ir Dump for a pass with raw lines
type PassDump = {
    header: string;
    lines: OutputLine[];
};
// Ir Dump for a pass with raw lines broken into affected functions (or "<loop>")
type SplitPassDump = {
    header: string;
    functions: Record<string, OutputLine[]>;
};
// Pass name with before / after dump
type Pass = {
    name: string;
    after: OutputLine[];
    before: OutputLine[];
};

export class LlvmPrintAfterAllParser {
    filters: RegExp[];
    lineFilters: RegExp[];
    irDumpAfterHeader: RegExp;
    defineLine: RegExp;
    label: RegExp;
    instruction: RegExp;

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
            /^(!\d+) = !{.*}/, // meta
            /^(![.A-Z_a-z-]+) = !{.*}/, // meta
            /^attributes #\d+ = { .+ }$/, // attributes directive
        ];
        this.lineFilters = [
            /,? ![\dA-Za-z]+((?=( {)?$))/, // debug annotation
            /,? #\d+((?=( {)?$))/, // attribute annotation
        ];

        this.irDumpAfterHeader = /^\*{3} (.+) \*{3}(;.+)?$/;
        this.defineLine = /^define .+ @(\w+)\(.+$/;
        this.label = /^\d+:(\s+;.+)?$/;
        this.instruction = /^\s+.+$/;
    }

    breakdownDump(dump: PassDump) {
        // break up down dumps for each pass into functions (or absence of functions in the case of loop passes)
        // we have three cases of ir dumps to consider:
        // - Most passes dump a single function
        // - Some passes dump every function
        // - Some loop passes only dump loops - these are challenging to deal with
        const pass: SplitPassDump = {
            header: dump.header,
            functions: {},
        };
        let func: {
            name: string;
            lines: OutputLine[];
        } | null = null;
        for (const line of dump.lines) {
            const match = line.text.match(this.defineLine);
            // function define line
            if (match) {
                // if the last function has not been closed...
                if (func !== null) {
                    throw 'Internal error during breakdownPass (1)';
                }
                func = {
                    name: match[1],
                    lines: [line], // include the current line
                };
            } else {
                // close function
                if (line.text.trim() === '}') {
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
                            // otherwise a loop
                            func = {
                                name: '<loop>',
                                lines: [], // current line will be included with the push below, no need to include here
                            };
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
                    throw 'Internal error during breakdownPass (5)';
                }
                pass.functions[func.name] = func.lines;
            } else {
                throw 'Internal error during breakdownPass (6)';
            }
        }
        return pass;
    }

    breakdownOutput(ir: OutputLine[]) {
        // break down output by "*** IR Dump After XYZ ***" markers
        const raw_passes: PassDump[] = [];
        let pass: PassDump | null = null;
        for (const line of ir) {
            const match = line.text.match(this.irDumpAfterHeader);
            if (match) {
                if (pass !== null) {
                    raw_passes.push(pass);
                }
                pass = {
                    header: match[1],
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
        //console.dir(raw_passes, {
        //    depth: 3
        //});
        // Further break down by functions in each dump
        const passDumps = raw_passes.map(this.breakdownDump.bind(this));
        //console.dir(passDumps, {
        //    depth: 5
        //});
        // Currently we have an array of passes with a map of functions altered in each pass
        // We want to transpose to a map of functions with an array of passes on the function
        const passDumpsByFunction: Record<string, PassDump[]> = {};
        // I'm assuming loop dumps should correspond to the previous function dumped
        let previousFunction: string | null = null;
        for (const pass of passDumps) {
            const {header, functions} = pass;
            const functionEntries = Object.entries(functions);
            for (const [function_name, lines] of functionEntries) {
                const name: string | null = function_name === '<loop>' ? previousFunction : function_name;
                assert(name !== null, 'Loop dump without preceding dump');
                if (!(name in passDumpsByFunction)) {
                    passDumpsByFunction[name] = [];
                }
                passDumpsByFunction[name].push({
                    header,
                    lines,
                });
            }
            if (functionEntries.length > 0) {
                throw 'Internal error during breakdownOutput (2)';
            } else if (functionEntries.length === 1) {
                const name = functionEntries[0][0];
                if (name !== '<loop>') {
                    previousFunction = name;
                }
            } else {
                previousFunction = null;
            }
        }
        //console.dir(passDumpsByFunction, {
        //    depth: 5
        //});
        // We have all the passes for each function, now we will go through each function and match the before/after
        // dumps
        const final_output: Record<string, Pass[]> = {};
        for (const [function_name, passDumps] of Object.entries(passDumpsByFunction)) {
            // I had a fantastic chunk2 method to iterate the passes in chunks of 2 but I've been foiled by an edge
            // case: At least the "Delete dead loops" may only print a before dump and no after dump
            const passes: Pass[] = [];
            // i incremented appropriately later
            for (let i = 0; i < passDumps.length; ) {
                const pass: Pass = {
                    name: '',
                    after: [],
                    before: [],
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
                passes.push(pass);
            }
            assert(!(function_name in final_output), 'xxx');
            final_output[function_name] = passes;
        }
        //console.dir(final_output, {
        //    depth: 6
        //});
        return final_output;
    }

    process(ir: {text: string}[], _: ParseFilters) {
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
