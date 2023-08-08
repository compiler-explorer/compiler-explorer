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

import {AsmResultLabel, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

import {AsmParser} from './asm-parser.js';

export class CC65AsmParser extends AsmParser {
    labelWithAsmRe: RegExp;
    labelAssignmentRe: RegExp;
    justAsmRe: RegExp;
    labelAddressRe: RegExp;
    directiveRe: RegExp;
    labelExtractRe: RegExp;

    constructor(compilerProps) {
        super(compilerProps);

        this.labelWithAsmRe = /(L[\dA-F]{4}):\s*(.*)/;
        this.labelAssignmentRe = /(L[\dA-F]{4})\s*:=\s(\$[\dA-F]{4})/;
        this.justAsmRe = /^\s{8}([.a-z]+\s*.*)/i;
        this.commentRe = /^;\s.*/;
        this.labelAddressRe = /L([\dA-F]{4})/;
        this.directiveRe = /^\./;
        this.labelExtractRe = /(L[\dA-F]{4})/;
    }

    extractLabels(asmtext: string, colOffset: number): AsmResultLabel[] | undefined {
        const match = asmtext.match(this.labelExtractRe);
        if (match) {
            const labelsInLine: AsmResultLabel[] = [];

            const label = match[1];

            const startCol = asmtext.indexOf(label);
            labelsInLine.push({
                name: label,
                range: {
                    startCol: startCol + colOffset,
                    endCol: startCol + colOffset + label.length,
                },
            });

            return labelsInLine;
        }

        return undefined;
    }

    override processBinaryAsm(asm, filters: ParseFiltersAndOutputOptions) {
        const result: ParsedAsmResultLine[] = [];
        const asmLines = asm.split('\n');

        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return {asm: [{text: asmLines[0], source: null}]};
        }

        const labelDefinitions: Record<string, number> = {};

        for (const line of asmLines) {
            let match = line.match(this.commentRe);
            if (match) {
                if (!filters.commentOnly) {
                    result.push({
                        text: line,
                    });
                }
                continue;
            }

            match = line.match(this.labelAssignmentRe);
            if (match) {
                const label = match[1];
                result.push({
                    text: label + ' := ' + match[2],
                });
                labelDefinitions[label] = result.length + 1;
                continue;
            }

            match = line.match(this.labelWithAsmRe);
            if (match) {
                const label = match[1];
                const asmtext = match[2];

                if (filters.directives) {
                    const directiveMatch = asmtext.match(this.directiveRe);
                    if (directiveMatch) continue;
                }

                const address = label.match(this.labelAddressRe);
                if (address) {
                    result.push({
                        text: label + ':',
                        address: parseInt(address[1], 16),
                    });
                } else {
                    result.push({
                        text: label + ':',
                    });
                }

                labelDefinitions[label] = result.length + 1;

                result.push({
                    text: '  ' + asmtext,
                    labels: this.extractLabels(asmtext, 3),
                });

                continue;
            }

            match = line.match(this.justAsmRe);
            if (match) {
                const asmtext = match[1];

                if (filters.directives) {
                    const directiveMatch = asmtext.match(this.directiveRe);
                    if (directiveMatch) continue;
                }

                result.push({
                    text: '  ' + asmtext,
                    labels: this.extractLabels(asmtext, 3),
                });

                continue;
            }
        }

        return {
            asm: result,
            labelDefinitions: labelDefinitions,
        };
    }
}
