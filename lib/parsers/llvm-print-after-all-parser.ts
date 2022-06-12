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

        this.irDumpAfterHeader = /^\*{3} (.+) \*{3}$/;
        this.defineLine = /^define .+ @(\w+)\(.+$/;
        this.label = /^\d+:(\s+;.+)?$/;
        this.instruction = /^\s+.+$/;
    }

    getOutput(ir: {text: string}[]) {
        return ir;
    }

    process(ir: {text: string}[], _: ParseFilters) {
        let last_item: string | null = null; // awful impurity to used to filter adjacent blank lines
        // Filter a lot of junk before processing
        const preprocessed_lines = ir
            .filter(line => this.filters.every(re => line.text.match(re) === null)) // apply filters
            .filter(line => {
                // Filter duplicate blank lines
                const b = line.text === '' && last_item === '';
                last_item = line.text;
                return !b;
            })
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

        return this.getOutput(preprocessed_lines);
    }
}
