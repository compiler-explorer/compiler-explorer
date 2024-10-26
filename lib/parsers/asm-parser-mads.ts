// Copyright (c) 2024, Compiler Explorer Authors
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
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {assert} from '../assert.js';
import {PropertyGetter} from '../properties.interfaces.js';
import * as utils from '../utils.js';

import {AsmParser, ParsingContext} from './asm-parser.js';
import {AsmRegex} from './asmregex.js';

export class MadsAsmParser extends AsmParser {
    protected lineWithoutAddress: RegExp;
    protected standAloneLabel: RegExp;
    protected asmOpcodeReWithInlineLabel: RegExp;
    protected varAssignment: RegExp;
    protected constAssignment: RegExp;

    constructor(compilerProps: PropertyGetter) {
        super(compilerProps);

        this.labelDef = /^([Ll]_\d*)$/;
        this.assignmentDef = /^([A-Z_a-z][\w$.]*)\s*=/;

        this.stdInLooking = /<stdin>|^-$|output\.[^/]+$|<source>/;

        // not really a proper file/line annotation, but it's all we have
        this.source6502Dbg = /^; optimize (?:OK|FAIL) \((.*)\), line = (\d*)/;
        this.source6502DbgEnd = /\s*\.ENDL/;

        this.lineWithoutAddress = /[\d ]{6} (.*)/;

        this.asmOpcodeRe = /^[\d ]{6} (?<address>[\dA-F]+)\s(?<opcodes>([\dA-F]{2} ?)+)\s*(?<disasm>.*)/;
        this.asmOpcodeReWithInlineLabel =
            /^[\d ]{6} (?<address>[\dA-F]+) (?<opcodes>([\dA-F]{2} ?)+)\t+(?<label>[A-Z]\w*)\t+(?<disasm>.*)/;
        this.standAloneLabel = /^[\d ]{6} ([\dA-F]{4})\t+([@A-Za-z]\w*)/;

        this.constAssignment = /^[\d ]{6} (= )([\dA-Z]{4})\t+(.*)\t(= .*)/;
        this.varAssignment = /^[\d ]{6} ([\dA-Z]{4})\t+(\.var )(.*)\t(= .*)/;

        this.commentOnly = /^[\d ]{6} \t*(; .*)/;

        this.lineRe = /^[\d ]{6} (.*)/;
    }

    override handleSource(context: ParsingContext, line: string) {}

    override handleStabs(context: ParsingContext, line: string) {}

    getAsmLineWithOpcodeReMatch(
        line: string,
        source: AsmResultSource | undefined | null,
        filters: ParseFiltersAndOutputOptions,
        matchGroups: {[key: string]: string},
    ): ParsedAsmResultLine {
        const labelsInLine: AsmResultLabel[] = [];

        const address = parseInt(matchGroups.address, 16);
        const opcodes = (matchGroups.opcodes || '').split(' ').filter(x => !!x);
        let text = '';
        if (matchGroups.label) {
            text = matchGroups.label.trim() + ': ';
        }
        const disassembly = ' ' + AsmRegex.filterAsmLine(matchGroups.disasm, filters);
        const destMatch = line.match(this.destRe);
        if (destMatch) {
            const labelName = destMatch[2];
            const startCol = disassembly.indexOf(labelName) + 1;
            labelsInLine.push({
                name: labelName,
                range: {
                    startCol: startCol,
                    endCol: startCol + labelName.length,
                },
            });
        }

        return {
            opcodes: opcodes,
            address: address,
            text: text + disassembly,
            source: source,
            labels: labelsInLine,
        };
    }

    override processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        let asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;
        const source: AsmResultSource | undefined | null = null;

        // Handle "error" documents.
        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return {
                asm: [{text: asmLines[0], source: null}],
            };
        }

        if (filters.preProcessBinaryAsmLines !== undefined) {
            asmLines = filters.preProcessBinaryAsmLines(asmLines);
        }

        const linePrefix = filters.trim ? ' ' : '  ';

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

            let match = line.match(this.asmOpcodeReWithInlineLabel);
            if (match) {
                assert(match.groups);

                labelDefinitions[match.groups.label] = asm.length;
                asm.push(this.getAsmLineWithOpcodeReMatch(line, source, filters, match.groups));

                continue;
            }

            match = line.match(this.asmOpcodeRe);
            if (match) {
                assert(match.groups);

                const parsedLine = this.getAsmLineWithOpcodeReMatch(line, source, filters, match.groups);
                parsedLine.text = linePrefix + parsedLine.text;
                asm.push(parsedLine);

                continue;
            }

            match = line.match(this.standAloneLabel);
            if (match) {
                const address = parseInt(match[1], 16);
                const label = match[2];
                labelDefinitions[label] = asm.length;
                asm.push({
                    address: address,
                    text: label + ':',
                });
                continue;
            }

            match = line.match(this.commentOnly);
            if (match) {
                if (!filters.commentOnly) {
                    asm.push({
                        text: linePrefix + match[1],
                    });
                }

                continue;
            }

            match = line.match(this.constAssignment);
            if (match) {
                // const value = parseInt(match[1], 16);

                const label = match[3];
                labelDefinitions[label] = asm.length;
                asm.push({
                    text: match[3] + ' ' + match[4],
                });
                continue;
            }

            match = line.match(this.varAssignment);
            if (match) {
                const address = parseInt(match[1], 16);
                const label = match[3];
                labelDefinitions[label] = asm.length;
                asm.push({
                    address: address,
                    text: match[2] + match[3] + ' ' + match[4],
                });
                continue;
            }

            if (!filters.directives) {
                match = line.match(this.lineRe);
                if (match) {
                    asm.push({
                        text: linePrefix + match[1],
                    });
                }
            }
        }

        this.removeLabelsWithoutDefinition(asm, labelDefinitions);

        const endTime = process.hrtime.bigint();

        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
            languageId: 'asm6502',
        };
    }
}
