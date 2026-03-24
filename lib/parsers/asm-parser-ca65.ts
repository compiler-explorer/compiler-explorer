// Copyright (c) 2026, Compiler Explorer Authors
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

import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {PropertyGetter} from '../properties.interfaces.js';
import {AsmParser} from './asm-parser.js';

export class AsmParserCa65 extends AsmParser {
    // ca65 listing format:
    // AAAAAA  D  [HH HH HH ...]  source text
    // where AAAAAA is hex address (may end with 'r' for relocatable),
    // D is include depth, and HH are hex bytes.
    // Lines may also be blank or contain just a page header.
    listingLineRe: RegExp;
    pageHeaderRe: RegExp;

    constructor(compilerProps: PropertyGetter) {
        super(compilerProps);

        // Match: address (6 hex + optional 'r'), depth, optional hex/reloc bytes, then source.
        // Hex bytes can include 'rr' (relocatable) and 'xx' (uninitialised) markers.
        this.listingLineRe = /^([0-9A-Fa-f]{6}r?)\s+(\d+)\s{2}((?:(?:[0-9A-Fa-f]{2}|rr|xx)\s)*)\s*(.*)/;
        // Page headers and file info lines from ca65 listing preamble
        this.pageHeaderRe = /^\f|^ca65\s|^Main file\s|^Current file:/;
    }

    override processBinaryAsm(asm: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const result: ParsedAsmResultLine[] = [];
        const asmLines = asm.split('\n');

        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return {asm: [{text: asmLines[0], source: null}]};
        }

        for (const line of asmLines) {
            if (line.trim() === '' || this.pageHeaderRe.test(line)) {
                continue;
            }

            const match = line.match(this.listingLineRe);
            if (match) {
                const address = match[1];
                const hexBytes = match[3].trim();
                const sourceText = match[4];

                if (!sourceText && !hexBytes) {
                    continue;
                }

                let text: string;
                if (hexBytes) {
                    const addr = address.replace(/r$/, '');
                    text = `${addr}: ${hexBytes.padEnd(12)} ${sourceText}`;
                } else {
                    text = '                     ' + sourceText;
                }

                result.push({text});
            } else {
                result.push({text: line});
            }
        }

        return {asm: result};
    }
}
