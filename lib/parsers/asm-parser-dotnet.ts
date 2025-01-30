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

import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import * as utils from '../utils.js';

import {IAsmParser} from './asm-parser.interfaces.js';

type InlineLabel = {name: string; range: {startCol: number; endCol: number}};
type Source = {file: string; line: number};

export class DotNetAsmParser implements IAsmParser {
    scanLabelsAndMethods(asmLines: string[], removeUnused: boolean) {
        const labelDef: Record<number, {name: string; remove: boolean}> = {};
        const methodDef: Record<number, string> = {};
        const labelUsage: Record<number, InlineLabel> = {};
        const methodUsage: Record<number, InlineLabel> = {};
        const allAvailable: string[] = [];
        const usedLabels: string[] = [];

        const methodRefRe = /^(call|jmp|tail\.jmp)\s+(.*)/g;
        const labelRefRe = /^\w+\s+.*?(G_M\w+)/g;
        const removeCommentAndWsRe = /^\s*(?<line>.*?)(\s*;.*)?\s*$/;

        for (const line in asmLines) {
            const trimmedLine = removeCommentAndWsRe.exec(asmLines[line])?.groups?.line;
            if (!trimmedLine || trimmedLine.startsWith(';')) continue;
            if (trimmedLine.endsWith(':')) {
                if (trimmedLine.includes('(')) {
                    let methodSignature = trimmedLine.substring(0, trimmedLine.length - 1);
                    if ((methodSignature.match(/\(/g) || []).length > 1) {
                        methodSignature = methodSignature.substring(0, methodSignature.lastIndexOf('(')).trimEnd();
                    }
                    methodDef[line] = methodSignature;
                    allAvailable.push(methodDef[line]);
                } else {
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
                    range: {startCol: index, endCol: index + name.length},
                };
                usedLabels.push(labelResult.value[1]);
            }

            const methodResult = trimmedLine.matchAll(methodRefRe).next();
            if (!methodResult.done) {
                let name = methodResult.value[2];
                if (name.startsWith('[')) {
                    if (name.endsWith(']')) {
                        // cases like `call [Foo]`, `jmp [Foo]`, `tail.jmp [Foo]`
                        name = name.substring(1, name.length - 1);
                    } else if (name.includes(']')) {
                        // cases like `tail.jmp [rax]Foo`
                        name = name.substring(name.indexOf(']') + 1);
                    }
                }
                const index = asmLines[line].indexOf(name) + 1;
                methodUsage[line] = {
                    name: name,
                    range: {startCol: index, endCol: index + name.length},
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

    cleanAsm(asmLines: string[]): string[] {
        const cleanedAsm: string[] = [];

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
            cleanedAsm.push(line);
        }

        return cleanedAsm;
    }

    process(asmResult: string, filters: ParseFiltersAndOutputOptions) {
        const startTime = process.hrtime.bigint();

        const asm: {
            text: string;
            source: Source | null;
            labels: InlineLabel[];
        }[] = [];
        let labelDefinitions: [string, number][] = [];

        let asmLines: string[] = this.cleanAsm(utils.splitLines(asmResult));
        const startingLineCount = asmLines.length;

        if (filters.commentOnly) {
            const commentRe = /^\s*(;.*)$/;
            asmLines = asmLines.flatMap(l => (commentRe.test(l) ? [] : [l]));
        }

        const result = this.scanLabelsAndMethods(asmLines, filters.labels!);

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
        labelDefinitions = labelDefinitions.sort((a, b) => (a[1] < b[1] ? -1 : 1));

        for (const index in labelDefinitions) {
            if (result.labelDef[labelDefinitions[index][1]] && result.labelDef[labelDefinitions[index][1]].remove) {
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
