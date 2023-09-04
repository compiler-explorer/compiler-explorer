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
import * as utils from '../utils.js';

import {AsmParser} from './asm-parser.js';

export class SPIRVAsmParser extends AsmParser {
    parseOpString(asmLines) {
        const opString = /^\s*%(\d+)\s+=\s+OpString\s+"([^"]+)"$/;
        const files = {};
        for (const line of asmLines) {
            const match = line.match(opString);
            if (match) {
                const lineNum = parseInt(match[1]);
                files[lineNum] = match[2];
            }
        }
        return files;
    }

    override getUsedLabelsInLine(line): AsmResultLabel[] {
        const labelsInLine: AsmResultLabel[] = [];

        const labelPatterns = [
            /OpFunctionCall\s+%\w+\s+(%\w+)/,
            /OpBranch\s+(%\w+)/,
            /OpBranchConditional\s+%\w+\s+(%\w+)\s+(%\w+)/,
            /OpSelectionMerge\s+(%\w+)/,
            /OpLoopMerge\s+(%\w+)\s+(%\w+)/,
        ];
        let labels: string[] = [];
        for (const pattern of labelPatterns) {
            const labelMatches = line.match(pattern);
            if (labelMatches) {
                labels = labels.concat(labelMatches.slice(1));
            }
        }

        const switchMatches = line.match(/OpSwitch\s+%\w+\s+(%\w+)((?:\s+\d+\s+%\w+)*)/);
        if (switchMatches) {
            // default case
            labels.push(switchMatches[1]);
            const cases = switchMatches[2];
            const caseMatches = cases.matchAll(/\d+\s+(%\w+)/g);
            for (const caseMatch of caseMatches) {
                labels.push(caseMatch[1]);
            }
        }

        for (const label of labels) {
            const startCol = line.indexOf(label) + 1;
            labelsInLine.push({
                name: label,
                range: {
                    startCol: startCol,
                    endCol: startCol + label.length,
                },
            });
        }

        return labelsInLine;
    }

    override processAsm(asmResult, filters) {
        const startTime = process.hrtime.bigint();

        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        let asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;
        if (filters.preProcessLines !== undefined) {
            asmLines = filters.preProcessLines(asmLines);
        }
        const opStrings = this.parseOpString(asmLines);

        const sourceTag = /^\s*OpLine\s+%(\d+)\s+(\d+)\s+(\d*)$/;
        const endBlock = /OpFunctionEnd/;
        const comment = /^;/;
        const opLine = /OpLine/;
        const opNoLine = /OpNoLine/;
        const opExtDbg = /OpExtInst\s+%void\s+%\d+\s+Debug/;
        const opString = /OpString/;
        const opSource = /OpSource/;
        const opName = /OpName/;

        const labelDef = /^\s*(%\w+)\s*=\s*(?:OpFunction\s+|OpLabel)/;

        const unclosedString = /^[^"]+"(?:[^\\"]|\\.)*$/;
        const closeQuote = /^(?:[^\\"]|\\.)*"/;
        let inString = false;

        let source: any = null;

        for (let line of asmLines) {
            if (inString) {
                if (closeQuote.test(line)) {
                    inString = false;
                }

                continue;
            }

            const match = line.match(sourceTag);
            if (match) {
                source = {
                    file: utils.maskRootdir(opStrings[parseInt(match[1])]),
                    line: parseInt(match[2]),
                    mainsource: true,
                };
                const sourceCol = parseInt(match[3]);
                if (!isNaN(sourceCol) && sourceCol !== 0) {
                    source.column = sourceCol;
                }
            }

            if (endBlock.test(line) || opNoLine.test(line)) {
                source = null;
            }

            if (filters.commentOnly && comment.test(line)) {
                continue;
            }
            if (filters.directives) {
                if (
                    opLine.test(line) ||
                    opExtDbg.test(line) ||
                    opNoLine.test(line) ||
                    opString.test(line) ||
                    opSource.test(line) ||
                    opName.test(line)
                ) {
                    if (unclosedString.test(line)) {
                        inString = true;
                    }
                    continue;
                }
            }

            line = utils.expandTabs(line);

            const labelDefMatch = line.match(labelDef);
            if (labelDefMatch) {
                labelDefinitions[labelDefMatch[1]] = asm.length + 1;
            }

            const labelsInLine = this.getUsedLabelsInLine(line);
            asm.push({
                text: line,
                source: source,
                labels: labelsInLine,
            });
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions,
            languageId: 'spirv',
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
