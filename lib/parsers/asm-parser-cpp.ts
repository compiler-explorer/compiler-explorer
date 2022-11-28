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

import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import * as utils from '../utils';

import {IAsmParser} from './asm-parser.interfaces';

type InlineLabel = {name: string; range: {startCol: number; endCol: number}};
type Source = {file: string | null; line: number};

const lineRe = /^\s*#line\s+(?<line>\d+)\s+"(?<file>[^"]+)"/;

export class AsmParserCpp implements IAsmParser {
    process(asmResult: string, filters: ParseFiltersAndOutputOptions) {
        const startTime = process.hrtime.bigint();

        const asm: {
            text: string;
            source: Source | null;
            labels: InlineLabel[];
        }[] = [];

        let source: Source | null = null;
        for (const line of utils.splitLines(asmResult)) {
            let advance = true;
            const match = line.match(lineRe);
            if (match && match.groups) {
                // TODO perhaps we'll need to check the file here at some point in the future.
                // TODO I've temporarily disabled this as the result is visually too noisy
                // was:  source = {file: null, line: parseInt(match.groups.line)};
                source = {file: match.groups.file, line: parseInt(match.groups.line)};
                if (filters.directives) {
                    continue;
                }
                advance = false;
            }
            asm.push({
                text: line,
                source: source,
                labels: [],
            });
            if (source && advance) {
                source = {...source, line: source.line + 1};
            }
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: [],
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: 0,
        };
    }
}
