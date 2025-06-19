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

import {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {PropertyGetter} from '../properties.interfaces.js';
import * as utils from '../utils.js';

import {AsmParser} from './asm-parser.js';

export class PTXAsmParser extends AsmParser {
    protected commentOnlyLine: RegExp;
    protected emptyLine: RegExp;

    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);

        // PTX comment patterns
        this.commentOnlyLine = /^\s*\/\//; // Line that starts with // (after optional whitespace)
        this.emptyLine = /^\s*$/; // Line with only whitespace
    }

    override processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();

        const asm: ParsedAsmResultLine[] = [];
        const asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;

        // Process each line and convert to ParsedAsmResultLine
        for (let line of asmLines) {
            // Apply comment filter
            if (filters.commentOnly) {
                // Skip lines that are only comments or empty
                if (this.commentOnlyLine.test(line) || this.emptyLine.test(line)) {
                    continue;
                }

                // Strip inline comments (// at end of line)
                const commentIndex = line.indexOf('//');
                if (commentIndex > 0) {
                    line = line.substring(0, commentIndex).trimEnd();
                }
            }

            asm.push({
                text: line,
                source: null,
                labels: [],
            });
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: {},
            languageId: 'ptx',
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
