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
import {PropertyGetter} from '../properties.interfaces.js';
import * as utils from '../utils.js';

import {AsmParser} from './asm-parser.js';

export class NorcroftObjAsmParser extends AsmParser {
    private readonly asmBinaryParser: AsmParser;
    private readonly headerOrTrailer = /^\s*(?:\S+\s+)?(AREA|END|EXPORT|IMPORT|NOINIT|BASED|COMDEF|COMMON|DATA)\b/;
    private readonly sourceFile = /^\s*;\s*source-file:\s+"([^"]+)",\s*line\s+(\d+)/;
    private readonly lineNumber = /^\s*;\s*line:\s+(\d+)(?:\s+\d+)?/;
    private readonly label = /^([|A-Za-z_.$][\w.$|]*)\s*$/; // label has no whitespace at start of line.
    private readonly DCD = /^\s*(?:\S+\s+)?(DC[A-Z]*)\b/; // DCD, DCB, DCFD, etc...

    constructor(compilerProps: PropertyGetter) {
        super(compilerProps);
        this.asmBinaryParser = new AsmParser(compilerProps);
    }

    override processAsm(asm: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        if (filters.binary) return this.asmBinaryParser.processBinaryAsm(asm, filters);

        let currentfile = '';
        let currentline: string | undefined;

        const asmLines: ParsedAsmResultLine[] = [];
        asm = asm.replace(/\u001A$/, '');

        utils.eachLine(asm, line => {
            let isDirective = false;

            const labelmatch = line.match(this.label);
            if (labelmatch && !labelmatch[1].startsWith('|')) {
                // global label (eg. "main", not "|L030|") - reset any line number.
                currentline = undefined;
            }

            // Header and trailer keywords should be white and reset any line number.
            if (line.match(this.headerOrTrailer)) {
                // possibly also current file?
                currentline = undefined;
                isDirective = true;
            }

            const sourcefilematch = line.match(this.sourceFile);
            if (sourcefilematch) {
                currentfile = sourcefilematch[1];
                currentline = sourcefilematch[2];
                isDirective = true;
            }

            const linenumbermatch = line.match(this.lineNumber);
            if (linenumbermatch) {
                currentline = linenumbermatch[1];
                isDirective = true;
            }

            // Parse the line number from any updated string.
            // Anything before here can parse or reset the current line.
            // Anything after can filter a line from display (return) or whiten
            // the line (set source to null).
            let source: AsmResultSource | null = null;
            if (currentfile && currentline) {
                source = {
                    file: filters.dontMaskFilenames ? currentfile : null,
                    line: Number.parseInt(currentline, 10),
                };
            }

            // Comments and blank lines should be filtered as directives.
            // Note that this includes sourcefilematch and linenumber match -
            // if the behaviour for them needs to change, check those bools.
            const trimmed = line.trim();
            if (trimmed === '' || trimmed.startsWith(';')) {
                isDirective = true;
                source = null; // also render white
            }

            if (labelmatch) {
                source = null; // render all labels white
            }

            if (line.match(this.DCD)) {
                source = null; // render all DCDs(etc) white
            }

            // Don't append directives if they're filtered out.
            if (filters.directives && isDirective) return;

            // All lines must be appended to appear in CE's output.
            asmLines.push({
                text: line,
                source,
            });
        });

        return {
            asm: asmLines,
        };
    }
}
