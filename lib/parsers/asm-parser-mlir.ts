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

import {AsmResultSource, ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import * as utils from '../utils.js';

import {AsmParser} from './asm-parser.js';

export class MlirAsmParser extends AsmParser {
    protected locDefRegex: RegExp;
    protected locDefUnknownRegex: RegExp;
    protected locRefRegex: RegExp;
    protected locRefRegexReplace: RegExp;
    protected inlineLocRegex: RegExp;
    protected inlineLocRegexReplace: RegExp;

    constructor() {
        super();

        // Match location definitions like #loc1 = loc("/path/to/file":line:column)
        this.locDefRegex = /^#(\w+)\s*=\s*loc\("([^"]+)":(\d+):(\d+)\)/;

        // Match location definitions like #loc1 = loc(unknown)
        this.locDefUnknownRegex = /^#(\w+)\s*=\s*loc\(unknown\)/;

        // Match location references like loc(#loc1)
        this.locRefRegex = /\s*loc\(#(\w+)\)/;
        this.locRefRegexReplace = new RegExp(this.locRefRegex.source, 'g');

        // Match inline locations like loc("/path/to/file":line:column)
        this.inlineLocRegex = /\s*loc\("([^"]+)":(\d+):(\d+)\)/;
        this.inlineLocRegexReplace = new RegExp(this.inlineLocRegex.source, 'g');
    }

    override processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();

        const asm: ParsedAsmResultLine[] = [];
        const asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;

        // First pass: extract all location definitions
        const locationMap = new Map<string, AsmResultSource>();
        for (const line of asmLines) {
            const locMatch = line.match(this.locDefRegex);
            if (locMatch) {
                const locId = locMatch[1];
                const file = locMatch[2];
                const lineNum = Number.parseInt(locMatch[3], 10);
                const column = Number.parseInt(locMatch[4], 10);

                locationMap.set(locId, {
                    file: utils.maskRootdir(file),
                    line: lineNum,
                    column: column,
                    mainsource: true,
                });
            }
        }

        // Second pass: process each line and associate with source information
        for (const line of asmLines) {
            // Skip location definition lines
            if (this.locDefRegex.test(line) || this.locDefUnknownRegex.test(line)) {
                continue;
            }

            // Apply filters if needed
            let processedLine = line;
            if (filters.trim) {
                processedLine = processedLine.trim();
            }

            if (filters.commentOnly && processedLine.trim().startsWith('//')) {
                continue;
            }

            // Find source information from location references
            let source: AsmResultSource | null = null;

            // Check for location references like loc(#loc1)
            const locRefMatch = line.match(this.locRefRegex);
            if (locRefMatch) {
                const locId = locRefMatch[1];
                source = locationMap.get(locId) || null;
                // Remove location reference from the displayed text
                processedLine = processedLine.replace(this.locRefRegexReplace, '');
            } else {
                // Check for inline locations like loc("/path/to/file":line:column)
                const inlineLocMatch = line.match(this.inlineLocRegex);
                if (inlineLocMatch) {
                    const file = inlineLocMatch[1];
                    const lineNum = Number.parseInt(inlineLocMatch[2], 10);
                    const column = Number.parseInt(inlineLocMatch[3], 10);

                    source = {
                        file: utils.maskRootdir(file),
                        line: lineNum,
                        column: column,
                        mainsource: true,
                    };
                }
                // Remove inline location from the displayed text
                processedLine = processedLine.replace(this.inlineLocRegexReplace, '');
            }

            // Add the line to the result
            asm.push({
                text: processedLine,
                source: source,
                labels: [],
            });
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: {},
            languageId: 'mlir',
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
