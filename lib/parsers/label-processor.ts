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

import _ from 'underscore';

import {AsmResultLabel, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {unwrap} from '../assert.js';

export type LabelContext = {
    hasOpcode: (line: string, inNvccCode?: boolean, inVLIWpacket?: boolean) => boolean;
    checkVLIWpacket: (line: string, inVLIWpacket: boolean) => boolean;
    labelDef: RegExp;
    dataDefn: RegExp;
    commentRe: RegExp;
    instructionRe: RegExp;
    identifierFindRe: RegExp;
    definesGlobal: RegExp;
    definesWeak: RegExp;
    definesAlias: RegExp;
    definesFunction: RegExp;
    cudaBeginDef: RegExp;
    startAppBlock: RegExp;
    endAppBlock: RegExp;
    startAsmNesting: RegExp;
    endAsmNesting: RegExp;
    mipsLabelDefinition: RegExp;
    labelFindNonMips: RegExp;
    labelFindMips: RegExp;
    fixLabelIndentation: (line: string) => string;
};

export class LabelProcessor {
    getLabelFind(asmLines: string[], context: LabelContext): RegExp {
        const isMips = _.any(asmLines, line => context.mipsLabelDefinition.test(line));
        return isMips ? context.labelFindMips : context.labelFindNonMips;
    }

    getUsedLabelsInLine(line: string, context: LabelContext): AsmResultLabel[] {
        const labelsInLine: AsmResultLabel[] = [];

        // Strip any comments
        const instruction = line.split(context.commentRe, 1)[0];

        // Remove the instruction
        const params = instruction.replace(context.instructionRe, '');

        const removedCol = instruction.length - params.length + 1;
        params.replace(context.identifierFindRe, (symbol, target, index) => {
            const startCol = removedCol + index;
            const label: AsmResultLabel = {
                name: symbol,
                range: {
                    startCol: startCol,
                    endCol: startCol + symbol.length,
                },
            };
            if (target !== symbol) {
                label.target = target;
            }
            labelsInLine.push(label);
            return symbol;
        });

        return labelsInLine;
    }

    removeLabelsWithoutDefinition(asm: ParsedAsmResultLine[], labelDefinitions: Record<string, number>) {
        for (const obj of asm) {
            if (obj.labels) {
                obj.labels = obj.labels.filter(label => labelDefinitions[label.target || label.name]);
            }
        }
    }

    findUsedLabels(asmLines: string[], filterDirectives: boolean, context: LabelContext): Set<string> {
        const labelsUsed: Set<string> = new Set();
        const weakUsages: Map<string, Set<string>> = new Map();

        function markWeak(fromLabel: string, toLabel: string) {
            if (!weakUsages.has(fromLabel)) weakUsages.set(fromLabel, new Set());
            unwrap(weakUsages.get(fromLabel)).add(toLabel);
        }

        const labelFind = this.getLabelFind(asmLines, context);
        let currentLabelSet: string[] = [];
        let inLabelGroup = false;
        let inCustomAssembly = 0;
        const startBlock = /\.cfi_startproc/;
        const endBlock = /\.cfi_endproc/;
        let inFunction = false;
        let inNvccCode = false;
        let inVLIWpacket = false;
        let definingAlias: string | undefined;

        for (let line of asmLines) {
            if (context.startAppBlock.test(line.trim()) || context.startAsmNesting.test(line.trim())) {
                inCustomAssembly++;
            } else if (context.endAppBlock.test(line.trim()) || context.endAsmNesting.test(line.trim())) {
                inCustomAssembly--;
            } else if (startBlock.test(line)) {
                inFunction = true;
            } else if (endBlock.test(line)) {
                inFunction = false;
            } else if (context.cudaBeginDef.test(line)) {
                inNvccCode = true;
            } else {
                inVLIWpacket = context.checkVLIWpacket(line, inVLIWpacket);
            }

            if (inCustomAssembly > 0) line = context.fixLabelIndentation(line);

            let match = line.match(context.labelDef);
            if (match) {
                if (inLabelGroup) currentLabelSet.push(match[1]);
                else currentLabelSet = [match[1]];
                inLabelGroup = true;
                if (definingAlias) {
                    markWeak(definingAlias, match[1]);
                }
            } else {
                if (inLabelGroup) {
                    inLabelGroup = false;
                    definingAlias = undefined;
                }
            }

            match = line.match(context.definesGlobal);
            if (!match) match = line.match(context.definesWeak);
            if (!match) match = line.match(context.cudaBeginDef);
            if (match) labelsUsed.add(match[1]);

            const definesAlias = line.match(context.definesAlias);
            if (definesAlias) {
                definingAlias = definesAlias[1];
            }

            const definesFunction = line.match(context.definesFunction);
            if (!definesFunction && (!line || line[0] === '.')) continue;

            match = line.match(labelFind);
            if (!match) continue;

            if (!filterDirectives || context.hasOpcode(line, inNvccCode, inVLIWpacket) || definesFunction) {
                for (const label of match) labelsUsed.add(label);
            } else {
                const isDataDefinition = context.dataDefn.test(line);
                const isOpcode = context.hasOpcode(line, inNvccCode, inVLIWpacket);
                if (isDataDefinition || isOpcode) {
                    if (inFunction && isDataDefinition) {
                        for (const label of match) labelsUsed.add(label);
                    } else {
                        for (const currentLabel of currentLabelSet) {
                            for (const label of match) markWeak(currentLabel, label);
                        }
                    }
                }
            }
        }

        const recurseMarkUsed = (label: string) => {
            labelsUsed.add(label);
            const usages = weakUsages.get(label);
            if (!usages) return;
            for (const nowUsed of usages) {
                if (!labelsUsed.has(nowUsed)) recurseMarkUsed(nowUsed);
            }
        };

        for (const label of new Set(labelsUsed)) recurseMarkUsed(label);
        return labelsUsed;
    }

    isLabelUsed(labelName: string, usedLabels: Set<string>, match: RegExpMatchArray, line: string): boolean {
        return usedLabels.has(labelName) && (match[0] !== line || (match[2] !== undefined && match[2].trim() !== '.'));
    }

    shouldFilterLabel(match: RegExpMatchArray, line: string, labelsUsed: Set<string>, filtersLabels: boolean): boolean {
        if (!filtersLabels) return false;

        return !labelsUsed.has(match[1]) && match[0] === line && (match[2] === undefined || match[2].trim() === '.');
    }
}
