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

import {AsmResultLabel, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {PropertyGetter} from '../properties.interfaces.js';

import {AsmParser} from './asm-parser.js';

export class ORCAAsmParser extends AsmParser {
    labelDefinitionRe: RegExp;
    operandRe: RegExp;
    labelInOperandRe: RegExp;
    directiveRe: RegExp;
    linePreCommentRe: RegExp;
    labelOpcodeOperandRe: RegExp;
    buggyLabelRe: RegExp;

    constructor(compilerProps: PropertyGetter) {
        super(compilerProps);

        this.commentRe = /^[!*;]\s.*/;
        this.labelDefinitionRe = /^([A-Z_a-z~][\w~]*)/;
        this.operandRe = /^((?:[A-Z_a-z~][\w~]*)?\s+[A-Za-z]{3}\s+)([^;]+)/;
        this.labelInOperandRe = /^[#(>|]*([A-Z_a-z~][\w~]*)/;
        this.directiveRe =
            /^\s+(?:align|anop|case|codechk|datachk|dynchk|eject|err|expand|ieee|instime|keep|kind|list|longa|longi|mem|merr|msb|numsex|objcase|printer|setcom|symbol|title|65c02|65816)\s+/;
        this.linePreCommentRe = /^(.{80}[^;]*);/;
        this.labelOpcodeOperandRe = /^((?:[A-Z_a-z~][\w~]*)?)\s+([\w~]+)\s*(.*)/;
        this.buggyLabelRe = /^([\w~]{20,})(start|entry)$/;
    }

    extractLabels(line: string): AsmResultLabel[] | undefined {
        let match = line.match(this.operandRe);
        if (match) {
            const preOperandLength = match[1].length;
            const operand = match[2];

            if (operand === 'a' || operand === 'A') {
                return undefined;
            }

            match = operand.match(this.labelInOperandRe);
            if (match) {
                const labelsInLine: AsmResultLabel[] = [];

                const label = match[1];

                const startCol = preOperandLength + operand.indexOf(label) + 1;
                labelsInLine.push({
                    name: label,
                    range: {
                        startCol: startCol,
                        endCol: startCol + label.length,
                    },
                });

                return labelsInLine;
            }
        }

        return undefined;
    }

    override processBinaryAsm(asm: string, filters: ParseFiltersAndOutputOptions) {
        const result: ParsedAsmResultLine[] = [];
        const asmLines = asm.split('\n');

        const labelDefinitions: Record<string, number> = {};

        for (let line of asmLines) {
            let match = line.match(this.commentRe);
            if (match) {
                if (!filters.commentOnly) {
                    result.push({
                        text: line,
                    });
                }
                continue;
            }

            // Insert missing spaces after long labels.
            // These may be omitted due to a bug in Golden Gate's dumpobj.
            match = line.match(this.buggyLabelRe);
            if (match) {
                line = match[1] + ' ' + match[2];
            }

            match = line.match(this.labelDefinitionRe);
            if (match) {
                labelDefinitions[match[1]] = result.length + 1;
            }

            if (filters.commentOnly || filters.trim) {
                match = line.match(this.linePreCommentRe);
                if (match) {
                    line = match[1];
                }
            }

            if (filters.directives) {
                match = line.toLowerCase().match(this.directiveRe);
                if (match) continue;
            }

            if (filters.trim) {
                match = line.match(this.labelOpcodeOperandRe);
                if (match) {
                    line = match[1] + (match[1].length === 0 ? '  ' : ' ') + match[2] + ' ' + match[3];
                }
            }

            if (result.length === 0 || (filters.trim && result[result.length - 1].text.trim() !== 'end')) {
                if (line.trim() === '') continue;
            }

            result.push({
                text: line.trimEnd(),
                labels: this.extractLabels(line),
            });
        }

        return {
            asm: result,
            labelDefinitions: labelDefinitions,
        };
    }
}
