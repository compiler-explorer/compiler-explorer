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

type MlirLocation =
    | {
          kind: 'source';
          source: AsmResultSource;
      }
    | {
          kind: 'alias';
          target: string;
      }
    | {
          kind: 'unknown';
      };

export class MlirAsmParser extends AsmParser {
    protected locDefRegex: RegExp;
    protected locDefUnknownRegex: RegExp;
    protected locDefAliasRegex: RegExp;
    protected locDefAnyRegex: RegExp;
    protected locRefRegex: RegExp;
    protected locRefRegexReplace: RegExp;
    protected namedLocRefRegex: RegExp;
    protected namedLocRefRegexReplace: RegExp;
    protected inlineLocRegex: RegExp;
    protected inlineLocRegexReplace: RegExp;
    protected unknownLocRegexReplace: RegExp;

    constructor() {
        super();

        // Match location definitions like #loc1 = loc("/path/to/file":line:column)
        this.locDefRegex = /^\s*#(\w+)\s*=\s*loc\("([^"]+)":(\d+):(\d+)\)/;

        // Match location definitions like #loc1 = loc(unknown)
        this.locDefUnknownRegex = /^\s*#(\w+)\s*=\s*loc\(unknown\)/;

        // Match location definitions like #loc1 = loc("name"(#loc2))
        this.locDefAliasRegex = /^\s*#(\w+)\s*=\s*loc\("[^"]+"\(#(\w+)\)\)/;

        // Match any location definition line, including nested named locations.
        this.locDefAnyRegex = /^\s*#\w+\s*=\s*loc\(/;

        // Match location references like loc(#loc1)
        this.locRefRegex = /\s*loc\(#(\w+)\)/;
        this.locRefRegexReplace = new RegExp(this.locRefRegex.source, 'g');

        // Match named location references like loc("pid"(#loc1))
        this.namedLocRefRegex = /\s*loc\("[^"]+"\(#(\w+)\)\)/;
        this.namedLocRefRegexReplace = new RegExp(this.namedLocRefRegex.source, 'g');

        // Match inline locations like loc("/path/to/file":line:column)
        this.inlineLocRegex = /\s*loc\("([^"]+)":(\d+):(\d+)\)/;
        this.inlineLocRegexReplace = new RegExp(this.inlineLocRegex.source, 'g');

        this.unknownLocRegexReplace = /\s*loc\(unknown\)/g;
    }

    private getSource(file: string, lineNum: string, column: string): AsmResultSource {
        return {
            file: utils.maskRootdir(file),
            line: Number.parseInt(lineNum, 10),
            column: Number.parseInt(column, 10),
            mainsource: true,
        };
    }

    private resolveLocation(
        locId: string,
        locationMap: Map<string, MlirLocation>,
        seenLocIds = new Set<string>(),
    ): AsmResultSource | null {
        if (seenLocIds.has(locId)) {
            return null;
        }
        seenLocIds.add(locId);

        const location = locationMap.get(locId);
        if (!location || location.kind === 'unknown') {
            return null;
        }
        if (location.kind === 'source') {
            return location.source;
        }

        return this.resolveLocation(location.target, locationMap, seenLocIds);
    }

    override processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();

        const asm: ParsedAsmResultLine[] = [];
        const asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;

        // First pass: extract all location definitions
        const locationMap = new Map<string, MlirLocation>();
        for (const line of asmLines) {
            const locMatch = line.match(this.locDefRegex);
            if (locMatch) {
                locationMap.set(locMatch[1], {
                    kind: 'source',
                    source: this.getSource(locMatch[2], locMatch[3], locMatch[4]),
                });
                continue;
            }

            const locAliasMatch = line.match(this.locDefAliasRegex);
            if (locAliasMatch) {
                locationMap.set(locAliasMatch[1], {
                    kind: 'alias',
                    target: locAliasMatch[2],
                });
                continue;
            }

            const locUnknownMatch = line.match(this.locDefUnknownRegex);
            if (locUnknownMatch) {
                locationMap.set(locUnknownMatch[1], {
                    kind: 'unknown',
                });
            }
        }

        // Second pass: process each line and associate with source information
        for (const line of asmLines) {
            // Skip location definition lines
            if (this.locDefAnyRegex.test(line)) {
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
                source = this.resolveLocation(locRefMatch[1], locationMap);
            } else {
                // Check for named location references like loc("pid"(#loc1))
                const namedLocRefMatch = line.match(this.namedLocRefRegex);
                if (namedLocRefMatch) {
                    source = this.resolveLocation(namedLocRefMatch[1], locationMap);
                } else {
                    // Check for inline locations like loc("/path/to/file":line:column)
                    const inlineLocMatch = line.match(this.inlineLocRegex);
                    if (inlineLocMatch) {
                        source = this.getSource(inlineLocMatch[1], inlineLocMatch[2], inlineLocMatch[3]);
                    }
                }
            }

            processedLine = processedLine
                .replace(this.namedLocRefRegexReplace, '')
                .replace(this.locRefRegexReplace, '')
                .replace(this.inlineLocRegexReplace, '')
                .replace(this.unknownLocRegexReplace, '');

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
