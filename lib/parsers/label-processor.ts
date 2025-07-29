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

class FindLabelsState {
    public labelsUsed = new Set<string>();
    public weakUsages = new Map<string, Set<string>>();
    public currentLabelSet: string[] = [];
    public inLabelGroup = false;
    public inCustomAssembly = 0;
    public inFunction = false;
    public inNvccCode = false;
    public inVLIWpacket = false;
    public definingAlias: string | undefined;

    markWeak(fromLabel: string, toLabel: string): void {
        const usageSet = this.weakUsages.get(fromLabel) ?? new Set<string>();
        if (!this.weakUsages.has(fromLabel)) this.weakUsages.set(fromLabel, usageSet);
        usageSet.add(toLabel);
    }

    enterLabelGroup(label: string): void {
        if (this.inLabelGroup) {
            this.currentLabelSet.push(label);
        } else {
            this.currentLabelSet = [label];
        }
        this.inLabelGroup = true;

        if (this.definingAlias) {
            this.markWeak(this.definingAlias, label);
        }
    }

    exitLabelGroup(): void {
        this.inLabelGroup = false;
        this.definingAlias = undefined;
    }
}

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
        const isMips = asmLines.some(line => context.mipsLabelDefinition.test(line));
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
            if (target !== symbol) label.target = target;
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

    private updateAssemblyContext(line: string, context: LabelContext, state: FindLabelsState): void {
        const startBlock = /\.cfi_startproc/;
        const endBlock = /\.cfi_endproc/;

        const trimmedLine = line.trim();
        if (context.startAppBlock.test(trimmedLine) || context.startAsmNesting.test(trimmedLine)) {
            state.inCustomAssembly++;
        } else if (context.endAppBlock.test(trimmedLine) || context.endAsmNesting.test(trimmedLine)) {
            state.inCustomAssembly--;
        } else if (startBlock.test(line)) {
            state.inFunction = true;
        } else if (endBlock.test(line)) {
            state.inFunction = false;
        } else if (context.cudaBeginDef.test(line)) {
            state.inNvccCode = true;
        } else {
            state.inVLIWpacket = context.checkVLIWpacket(line, state.inVLIWpacket);
        }
    }

    private preprocessLine(originalLine: string, context: LabelContext, state: FindLabelsState): string {
        return state.inCustomAssembly > 0 ? context.fixLabelIndentation(originalLine) : originalLine;
    }

    private processLabelDefinition(line: string, context: LabelContext, state: FindLabelsState): void {
        const match = line.match(context.labelDef);
        if (match) {
            state.enterLabelGroup(match[1]);
        } else if (state.inLabelGroup) state.exitLabelGroup();
    }

    private processGlobalWeakDefinitions(line: string, context: LabelContext, state: FindLabelsState): void {
        const match =
            line.match(context.definesGlobal) ?? line.match(context.definesWeak) ?? line.match(context.cudaBeginDef);
        if (match) state.labelsUsed.add(match[1]);

        const definesAlias = line.match(context.definesAlias);
        if (definesAlias) state.definingAlias = definesAlias[1];
    }

    private processLabelUsages(
        line: string,
        context: LabelContext,
        state: FindLabelsState,
        filterDirectives: boolean,
        labelFind: RegExp,
    ): void {
        const definesFunction = line.match(context.definesFunction);
        if (!definesFunction && (!line || line[0] === '.')) return;

        const match = line.match(labelFind);
        if (!match) return;

        if (!filterDirectives || context.hasOpcode(line, state.inNvccCode, state.inVLIWpacket) || definesFunction) {
            for (const label of match) state.labelsUsed.add(label);
        } else {
            const isDataDefinition = context.dataDefn.test(line);
            const isOpcode = context.hasOpcode(line, state.inNvccCode, state.inVLIWpacket);
            if (isDataDefinition || isOpcode) {
                if (state.inFunction && isDataDefinition) {
                    for (const label of match) state.labelsUsed.add(label);
                } else {
                    for (const currentLabel of state.currentLabelSet) {
                        for (const label of match) state.markWeak(currentLabel, label);
                    }
                }
            }
        }
    }

    private resolveWeakUsages(state: FindLabelsState): void {
        const recurseMarkUsed = (label: string) => {
            state.labelsUsed.add(label);
            const usages = state.weakUsages.get(label);
            if (!usages) return;
            for (const nowUsed of usages) {
                if (!state.labelsUsed.has(nowUsed)) recurseMarkUsed(nowUsed);
            }
        };

        // Create a snapshot of labelsUsed to avoid processing labels added during recursion
        for (const label of new Set(state.labelsUsed)) recurseMarkUsed(label);
    }

    findUsedLabels(asmLines: string[], filterDirectives: boolean, context: LabelContext): Set<string> {
        const state = new FindLabelsState();
        const labelFind = this.getLabelFind(asmLines, context);

        for (const originalLine of asmLines) {
            this.updateAssemblyContext(originalLine, context, state);
            const line = this.preprocessLine(originalLine, context, state);

            this.processLabelDefinition(line, context, state);
            this.processGlobalWeakDefinitions(line, context, state);
            this.processLabelUsages(line, context, state, filterDirectives, labelFind);
        }

        this.resolveWeakUsages(state);
        return state.labelsUsed;
    }

    isLabelUsed(labelName: string, usedLabels: Set<string>, match: RegExpMatchArray, line: string): boolean {
        return usedLabels.has(labelName) && (match[0] !== line || (match[2] !== undefined && match[2].trim() !== '.'));
    }

    shouldFilterLabel(match: RegExpMatchArray, line: string, labelsUsed: Set<string>, filtersLabels: boolean): boolean {
        if (!filtersLabels) return false;

        return !labelsUsed.has(match[1]) && match[0] === line && (match[2] === undefined || match[2].trim() === '.');
    }
}
