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

import {
    AsmResultLabel,
    AsmResultSource,
    ParsedAsmResult,
    ParsedAsmResultLine,
} from '../../types/asmresult/asmresult.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {assert} from '../assert.js';
import {PropertyGetter} from '../properties.interfaces.js';
import * as utils from '../utils.js';
import {AsmParser} from './asm-parser.js';
import {AsmRegex} from './asmregex.js';

export class E2KAsmParser extends AsmParser {
    private asmAddressE2K: RegExp;
    private asmInstructionE2K: RegExp;

    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);

        this.commentRe = /[#;!]/;

        // Lines matching the following pattern are considered comments:
        // - starts with '#', '@', '!', '//' or a single ';' (non repeated)
        // - starts with ';;' and the first non-whitespace before end of line is not #
        this.commentOnly = /^\s*(((#|@|!|\/\/).*)|(\/\*.*\*\/)|(;\s*)|(;[^;].*)|(;;\s*[^\s#].*))$/;

        this.asmAddressE2K = /^\s*([\da-f]+):\s*$/;
        this.asmInstructionE2K = /^ {2}\w/;
    }

    override filterAsmLine(line: string, filters: ParseFiltersAndOutputOptions): {text: string; skipLine: boolean} {
        // Default tabs expand is too wide for the first padding tab.
        if (line.startsWith('\t')) {
            line = '  ' + line.substring(1);
        }

        return super.filterAsmLine(line, filters);
    }

    override processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};
        const dontMaskFilenames = filters.dontMaskFilenames;

        let asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;
        let source: AsmResultSource | undefined | null = null;
        let func: string | null = null;
        let mayRemovePreviousLabel = true;
        let bundleAddress = 0;

        // Handle "error" documents.
        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return {
                asm: [{text: asmLines[0], source: null}],
            };
        }

        if (filters.preProcessBinaryAsmLines) asmLines = filters.preProcessBinaryAsmLines(asmLines);

        for (const line of asmLines) {
            const labelsInLine: AsmResultLabel[] = [];

            if (asm.length >= this.maxAsmLines) {
                if (asm.length === this.maxAsmLines) {
                    asm.push({
                        text: '[truncated; too many lines]',
                        source: null,
                        labels: labelsInLine,
                    });
                }
                continue;
            }
            let match = line.match(this.lineRe);
            if (match) {
                assert(match.groups);
                if (dontMaskFilenames) {
                    source = {
                        file: utils.maskRootdir(match[1]),
                        line: Number.parseInt(match.groups.line, 10),
                        mainsource: true,
                    };
                } else {
                    source = {file: null, line: Number.parseInt(match.groups.line, 10), mainsource: true};
                }
                continue;
            }

            // Extract current bundle address.
            match = line.match(this.asmAddressE2K);
            if (match) {
                const address = Number.parseInt(match[1], 16);
                if (!Number.isNaN(address)) {
                    bundleAddress = address;
                    continue;
                }
            }

            match = line.match(this.labelRe);
            if (match) {
                func = match[2];
                if (func && this.isUserFunction(func)) {
                    asm.push({
                        text: func + ':',
                        source: null,
                        labels: labelsInLine,
                    });
                    labelDefinitions[func] = asm.length;
                    if (process.platform === 'win32') source = null;
                }
                continue;
            }

            if (func && line === `${func}():`) continue;

            if (!func || !this.isUserFunction(func)) continue;

            // note: normally the source.file will be null if it's code from example.ext but with
            //  filters.dontMaskFilenames it will be filled with the actual filename instead we can test
            //  source.mainsource in that situation
            const isMainsource = source && (source.file === null || source.mainsource);
            if (filters.libraryCode && !isMainsource) {
                if (mayRemovePreviousLabel && asm.length > 0) {
                    const lastLine = asm[asm.length - 1];
                    if (lastLine.text && this.labelDef.test(lastLine.text)) {
                        asm.pop();
                    }
                    mayRemovePreviousLabel = false;
                }
                continue;
            }
            mayRemovePreviousLabel = true;

            match = line.match(this.relocationRe);
            if (match) {
                assert(match.groups);
                const address = Number.parseInt(match.groups.address, 16);
                const relocname = match.groups.relocname;
                const relocdata = match.groups.relocdata;
                // value/addend matched but not used yet.
                // const match_value = relocdata.match(this.relocDataSymNameRe);
                asm.push({
                    text: `  ${relocname} ${relocdata}`,
                    address: address,
                });
                continue;
            }

            if (this.asmInstructionE2K.test(line)) {
                const disassembly = AsmRegex.filterAsmLine(line, filters);
                asm.push({
                    // Disassembler does not output opcode bytes.
                    address: bundleAddress,
                    text: disassembly,
                    source: source,
                    labels: labelsInLine,
                });
                continue;
            }

            if (line === '') {
                asm.push({text: ''});
            }
        }

        this.removeLabelsWithoutDefinition(asm, labelDefinitions);

        const endTime = process.hrtime.bigint();

        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }
}

export class MCSTLCCE2KAsmParser extends E2KAsmParser {
    private sourceE2K: RegExp;

    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);

        // Match label define with a file:line comment.
        this.labelDef = /^(?:.proc\s+)?([\w$.@]+|"[\w$.@]+"):(?:\s*!.*$)?/i;
        this.sourceE2K = /!\s*[^:=]+\s*:\s*\d+\s*$/;
    }

    override filterAsmLine(line: string, filters: ParseFiltersAndOutputOptions): {text: string; skipLine: boolean} {
        const res = super.filterAsmLine(line, filters);
        if (res.skipLine) {
            return res;
        }
        let text = res.text;

        // These comments required for source line detection and cannot be
        // stripped earlier.
        if (filters.commentOnly && this.sourceE2K.test(text)) {
            const commentIndex = text.indexOf('!');
            if (commentIndex > 0) {
                text = text.substring(0, commentIndex).trimEnd();
                if (text === '') {
                    // The line contains only source comment. Skip to reduce
                    // the height of text.
                    return {text, skipLine: true};
                }
            }
        }

        return {text, skipLine: false};
    }
}
