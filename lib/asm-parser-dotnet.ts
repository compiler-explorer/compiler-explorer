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

import * as utils from './utils';

type InlineLabel = { name: string, range: { startCol: number, endCol: number } };
type Source = { file: string, line: number };

export class DotNetAsmParser {
    scanLabelsAndMethods(asmLines: string[], removeUnused: boolean) {
        const labelDef: Record<number, { name: string, remove: boolean }> = {};
        const methodDef: Record<number, string> = {};
        const labelUsage: Record<number, InlineLabel> = {};
        const methodUsage: Record<number, InlineLabel> = {};
        const allAvailable: string[] = [];
        const usedLabels: string[] = [];

        const methodRefRe = /^\w+\s+\[(.*)]/g;
        const tailCallRe = /^tail\.jmp\s+\[.*?](.*)/g;
        const labelRefRe = /^\w+\s+.*?(G_M\w+)/g;

        for (const line in asmLines) {
            const trimmedLine = asmLines[line].trim();
            if (!trimmedLine || trimmedLine.startsWith(';')) continue;
            if (trimmedLine.endsWith(':')) {
                if (trimmedLine.includes('(')) {
                    methodDef[line] = trimmedLine.substring(0, trimmedLine.length - 1);
                    allAvailable.push(methodDef[line]);
                }
                else {
                    labelDef[line] = {
                        name: trimmedLine.substring(0, trimmedLine.length - 1),
                        remove: false,
                    };
                    allAvailable.push(labelDef[line].name);
                }
                continue;
            }

            const labelResult = trimmedLine.matchAll(labelRefRe).next();
            if (!labelResult.done) {
                const name = labelResult.value[1];
                const index = asmLines[line].indexOf(name) + 1;
                labelUsage[line] = {
                    name: labelResult.value[1],
                    range: { startCol: index, endCol: index + name.length },
                };
                usedLabels.push(labelResult.value[1]);
            }

            let methodResult = trimmedLine.matchAll(methodRefRe).next();
            if (methodResult.done) methodResult = trimmedLine.matchAll(tailCallRe).next();
            if (!methodResult.done) {
                const name = methodResult.value[1];
                const index = asmLines[line].indexOf(name) + 1;
                methodUsage[line] = {
                    name: methodResult.value[1],
                    range: { startCol: index, endCol: index + name.length },
                };
            }
        }

        if (removeUnused) {
            for (const line in labelDef) {
                if (!usedLabels.includes(labelDef[line].name)) {
                    labelDef[line].remove = true;
                }
            }
        }

        return {
            labelDef,
            labelUsage,
            methodDef,
            methodUsage,
            allAvailable,
        };
    }

    cleanAsm(asmLines: string[]) {
        const cleanedAsm = [];

        for (const line of asmLines) {
            if (!line) continue;
            if (line.startsWith('; Assembly listing for method')) {
                if (cleanedAsm.length > 0) cleanedAsm.push('');
                // ; Assembly listing for method ConsoleApplication.Program:Main(System.String[])
                //                               ^ This character is the 31st character in this string.
                // `substring` removes the first 30 characters from it and uses the rest as a label.
                cleanedAsm.push(line.substring(30) + ':');
                continue;
            }

            if (line.startsWith('Emitting R2R PE file')) continue;
            if (line.startsWith(';') && !line.startsWith('; Emitting')) continue;

            cleanedAsm.push(line);
        }

        return cleanedAsm;
    }

    process(asmResult: string, filters) {
        const startTime = process.hrtime.bigint();

        const asm: {
            text: string,
            source: Source | null,
            labels: InlineLabel[],
        }[] = [];
        let labelDefinitions: [string, number][] = [];

        let asmLines: string[] = this.cleanAsm(utils.splitLines(asmResult));
        const startingLineCount = asmLines.length;

        if (filters.commentOnly) {
            const commentRe = /^\s*(;.*)$/g;
            asmLines = asmLines.flatMap(l => commentRe.test(l) ? [] : [l]);
        }

        const result = this.scanLabelsAndMethods(asmLines, filters.labels);

        for (const i in result.labelDef) {
            const label = result.labelDef[i];
            labelDefinitions.push([label.name, parseInt(i)]);
        }

        for (const i in result.methodDef) {
            const method = result.methodDef[i];
            labelDefinitions.push([method, parseInt(i)]);
        }

        for (const line in asmLines) {
            if (result.labelDef[line] && result.labelDef[line].remove) continue;

            const labels: InlineLabel[] = [];
            const label = result.labelUsage[line] || result.methodUsage[line];
            if (label) {
                if (result.allAvailable.includes(label.name)) {
                    labels.push(label);
                }
            }

            asm.push({
                text: asmLines[line],
                source: null,
                labels,
            });
        }

        let lineOffset = 1;
        labelDefinitions = labelDefinitions.sort((a, b) => a[1] < b[1] ? -1 : 1);

        for (const index in labelDefinitions) {
            if (result.labelDef[labelDefinitions[index][1]] &&
                result.labelDef[labelDefinitions[index][1]].remove) {
                labelDefinitions[index][1] = -1;
                lineOffset--;
                continue;
            }

            labelDefinitions[index][1] += lineOffset;
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: Object.fromEntries(labelDefinitions.filter(i => i[1] !== -1)),
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
