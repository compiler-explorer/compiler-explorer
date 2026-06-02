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

import {AsmResultSource, ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import * as utils from '../utils.js';
import {IAsmParser} from './asm-parser.interfaces.js';
import type {DotNetMethodKey, DotNetMethodSourceMapping, DotNetSourceMapping} from './pdb-parser-dotnet.js';
import {getCanonicalTypeSignature} from './pdb-parser-dotnet.js';

type InlineLabel = {name: string; range: {startCol: number; endCol: number}};
type ScanLabelsAndMethodsResult = {
    labelDef: Record<number, {name: string; remove: boolean}>;
    labelUsage: Record<number, InlineLabel>;
    methodDef: Record<number, string>;
    methodUsage: Record<number, InlineLabel>;
    allAvailable: Set<string>;
};

function formatMethodKey(method: DotNetMethodKey) {
    const typeName =
        method.typeArguments.length === 0 ? method.typeName : `${method.typeName}[${method.typeArguments.join(',')}]`;
    const methodKeyName =
        method.methodArguments.length === 0
            ? method.methodName
            : `${method.methodName}[${method.methodArguments.join(',')}]`;

    return `${typeName}:${methodKeyName}(${method.parameters.join(',')})${
        method.returnType && method.returnType !== 'void' ? `:${method.returnType}` : ''
    }`;
}

function parseJitDisasmMethodKey(methodName: string): DotNetMethodKey | null {
    const methodKeyText = methodName
        .trim()
        .replace(/:this$/, '')
        .replace(/::/g, ':');
    const methodMatch = methodKeyText.match(/^(?<typeAndMethod>.+)\((?<parameters>.*)\)(?::(?<returnType>.*))?$/);
    if (!methodMatch?.groups) {
        return null;
    }

    const methodSeparator = methodMatch.groups.typeAndMethod.lastIndexOf(':');
    if (methodSeparator === -1) {
        return null;
    }

    const type = parseJitGenericName(methodMatch.groups.typeAndMethod.substring(0, methodSeparator), '!');
    const method = parseJitGenericName(methodMatch.groups.typeAndMethod.substring(methodSeparator + 1), '!!');

    return {
        typeName: type.name,
        typeArguments: type.arguments.map(canonicalizeJitTypeName),
        methodName: method.name,
        methodArguments: method.arguments.map(canonicalizeJitTypeName),
        parameters: splitParameters(methodMatch.groups.parameters).map(canonicalizeJitTypeName),
        returnType: methodMatch.groups.returnType ? canonicalizeJitTypeName(methodMatch.groups.returnType) : '',
    };
}

function genericArgumentEnd(text: string, start: number) {
    let depth = 0;

    for (let index = start; index < text.length; index++) {
        if (text[index] === '[') {
            depth++;
        } else if (text[index] === ']') {
            depth--;
            if (depth === 0) {
                return index;
            }
        }
    }

    return -1;
}

function parseJitGenericName(name: string, genericParameterPrefix: '!' | '!!') {
    let canonical = '';
    const genericArguments: string[] = [];

    for (let index = 0; index < name.length; index++) {
        const char = name[index];
        if (char === '[' && name[index + 1] !== ']') {
            const end = genericArgumentEnd(name, index);
            if (end !== -1) {
                genericArguments.push(...splitParameters(name.substring(index + 1, end)));
                index = end;
                continue;
            }
        }

        canonical += char;
    }

    if (genericArguments.length === 0) {
        let genericParameterIndex = 0;
        for (const arity of canonical.matchAll(/`(\d+)/g)) {
            const remainingArity = 1024 * 1024 - genericParameterIndex;
            if (remainingArity <= 0) {
                break;
            }

            const arityToAdd = Math.min(Number.parseInt(arity[1], 10), remainingArity);
            for (let index = 0; index < arityToAdd; index++, genericParameterIndex++) {
                genericArguments.push(`${genericParameterPrefix}${genericParameterIndex}`);
            }
        }
    }

    return {name: canonical.replace(/`\d+/g, ''), arguments: genericArguments};
}

function canonicalizeJitTypeName(typeName: string) {
    const suffix = typeName.match(/(?:\[[,\s]*\]|\*|&)+$/)?.[0] ?? '';
    const typeWithoutSuffix = suffix ? typeName.substring(0, typeName.length - suffix.length) : typeName;

    let canonical = '';
    for (let index = 0; index < typeWithoutSuffix.length; index++) {
        const char = typeWithoutSuffix[index];
        if (char === '[' && typeWithoutSuffix[index + 1] !== ']') {
            const end = genericArgumentEnd(typeWithoutSuffix, index);
            if (end !== -1) {
                canonical += `[${splitParameters(typeWithoutSuffix.substring(index + 1, end))
                    .map(canonicalizeJitTypeName)
                    .join(',')}]`;
                index = end;
                continue;
            }
        }

        canonical += char;
    }

    return `${canonical.replace(/`\d+/g, '')}${suffix}`;
}

function substituteMetadataGenericParameters(typeName: string, requestedMethod: DotNetMethodKey) {
    return typeName.replace(/!!(\d+)|!(\d+)/g, (genericParameter, methodIndex, typeIndex) => {
        if (methodIndex !== undefined) {
            return requestedMethod.methodArguments[Number.parseInt(methodIndex, 10)] ?? genericParameter;
        }
        return requestedMethod.typeArguments[Number.parseInt(typeIndex, 10)] ?? genericParameter;
    });
}

function splitParameters(parameters: string) {
    const result: string[] = [];
    let depth = 0;
    let current = '';

    for (const char of parameters) {
        if (char === '[' || char === '(') {
            depth++;
        } else if ((char === ']' || char === ')') && depth > 0) {
            depth--;
        }

        if (char === ',' && depth === 0) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }

    if (current.trim()) {
        result.push(current.trim());
    }

    return result;
}

export class DotNetAsmParser implements IAsmParser {
    private readonly sourceMapping: DotNetSourceMapping;
    private readonly sourceMappingByName: Map<string, DotNetMethodSourceMapping>;
    private readonly sourceMappingOffsetsByMethod = new Map<DotNetMethodSourceMapping, number[]>();

    constructor(sourceMapping: DotNetSourceMapping = []) {
        this.sourceMapping = sourceMapping;
        this.sourceMappingByName = new Map(
            sourceMapping.map(methodSourceMapping => [
                formatMethodKey(methodSourceMapping.method),
                methodSourceMapping,
            ]),
        );
    }

    private getMethodSourceMapping(methodName: string) {
        const requestedMethod = parseJitDisasmMethodKey(methodName);
        if (!requestedMethod) {
            return undefined;
        }

        const exactMapping = this.sourceMappingByName.get(formatMethodKey(requestedMethod));
        if (exactMapping) {
            return exactMapping;
        }

        for (const methodSourceMapping of this.sourceMapping) {
            const candidate = methodSourceMapping.method;
            if (
                candidate.typeName !== requestedMethod.typeName ||
                candidate.methodName !== requestedMethod.methodName ||
                candidate.typeArguments.length !== requestedMethod.typeArguments.length ||
                candidate.methodArguments.length !== requestedMethod.methodArguments.length ||
                candidate.parameters.length !== requestedMethod.parameters.length
            ) {
                continue;
            }

            const candidateParameters =
                candidate.parameterTypes?.map(parameter => [
                    getCanonicalTypeSignature(parameter, requestedMethod).text,
                    getCanonicalTypeSignature(parameter, requestedMethod, false, false).text,
                ]) ??
                candidate.parameters.map(parameter => [
                    substituteMetadataGenericParameters(parameter, requestedMethod),
                ]);

            if (
                candidateParameters.every((parameters, index) => parameters.includes(requestedMethod.parameters[index]))
            ) {
                return methodSourceMapping;
            }
        }

        return undefined;
    }

    private getSourceMappingOffsets(methodSourceMapping: DotNetMethodSourceMapping) {
        let offsets = this.sourceMappingOffsetsByMethod.get(methodSourceMapping);
        if (offsets === undefined) {
            offsets = Object.keys(methodSourceMapping.offsets)
                .map(Number)
                .sort((a, b) => a - b);
            this.sourceMappingOffsetsByMethod.set(methodSourceMapping, offsets);
        }

        return offsets;
    }

    private getSourceForOffset(methodSourceMapping: DotNetMethodSourceMapping | undefined, offset: number) {
        if (!methodSourceMapping) {
            return null;
        }

        if (methodSourceMapping.offsets[offset] !== undefined) {
            return methodSourceMapping.offsets[offset];
        }

        const offsets = this.getSourceMappingOffsets(methodSourceMapping);
        let left = 0;
        let right = offsets.length - 1;
        let bestIndex = -1;
        while (left <= right) {
            const middle = Math.floor((left + right) / 2);
            if (offsets[middle] > offset) {
                right = middle - 1;
            } else {
                bestIndex = middle;
                left = middle + 1;
            }
        }

        return bestIndex === -1 ? null : methodSourceMapping.offsets[offsets[bestIndex]];
    }

    private computeSourceMappingForAsmLines(asmLines: string[], result: ScanLabelsAndMethodsResult) {
        const sources: Array<AsmResultSource | null> = [];
        let currentMethodSourceMapping: DotNetMethodSourceMapping | undefined;
        let currentSource: AsmResultSource | null = null;
        const inlineDebugInfoRe = /\bINL(?:RT|\d+)\s+@\s*(?:0[xX][0-9a-fA-F]+|\?\?\?)/;
        const inlineRootOffsetRe = /\bINLRT\s+@\s*0[xX](?<offset>[0-9a-fA-F]+)/g;

        for (const line in asmLines) {
            if (result.methodDef[line]) {
                currentMethodSourceMapping = this.getMethodSourceMapping(result.methodDef[line]);
                currentSource = null;
            }

            let lineSource: AsmResultSource | null = null;
            if (result.labelDef[line]) {
                // A JIT label starts a new native block. Require a fresh debug-info anchor
                // before mapping instructions in that block to avoid source leaking across blocks.
                currentSource = null;
            }

            let inlineRootOffset: string | undefined;
            for (const inlineRootOffsetMatch of asmLines[line].matchAll(inlineRootOffsetRe)) {
                inlineRootOffset = inlineRootOffsetMatch.groups?.offset;
            }
            if (inlineRootOffset !== undefined) {
                lineSource = null;
                // Inline chains may contain unknown child offsets, but a numeric INLRT still anchors the
                // instruction to the root method's IL offset.
                currentSource = this.getSourceForOffset(
                    currentMethodSourceMapping,
                    Number.parseInt(inlineRootOffset, 16),
                );
            } else if (inlineDebugInfoRe.test(asmLines[line])) {
                // The IL location is unknown, so we can't get a precise source mapping for this instruction.
                currentSource = null;
            }

            const trimmedAsmLine = asmLines[line].trimStart();
            if (inlineRootOffset === undefined && trimmedAsmLine !== '' && !trimmedAsmLine.startsWith(';')) {
                lineSource = currentSource;
            }

            sources.push(lineSource);
        }

        return sources;
    }

    scanLabelsAndMethods(asmLines: string[], removeUnused: boolean) {
        const labelDef: Record<number, {name: string; remove: boolean}> = {};
        const methodDef: Record<number, string> = {};
        const labelUsage: Record<number, InlineLabel> = {};
        const methodUsage: Record<number, InlineLabel> = {};
        const allAvailable = new Set<string>();
        const usedLabels = new Set<string>();

        const methodRefRe = /^(call|jmp|tail\.jmp)\s+(.*)/;
        const labelRefRe = /^\w+\s+.*?(G_M\w+)/;
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
                    allAvailable.add(methodDef[line]);
                } else {
                    labelDef[line] = {
                        name: trimmedLine.substring(0, trimmedLine.length - 1),
                        remove: false,
                    };
                    allAvailable.add(labelDef[line].name);
                }
                continue;
            }

            const labelResult = trimmedLine.match(labelRefRe);
            if (labelResult) {
                const name = labelResult[1];
                const index = asmLines[line].indexOf(name) + 1;
                labelUsage[line] = {
                    name,
                    range: {startCol: index, endCol: index + name.length},
                };
                usedLabels.add(name);
            }

            const methodResult = trimmedLine.match(methodRefRe);
            if (methodResult) {
                let name = methodResult[2];
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
                if (!usedLabels.has(labelDef[line].name)) {
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

            if (line.startsWith('Emitting R2R ')) continue;
            cleanedAsm.push(line);
        }

        return cleanedAsm;
    }

    process(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();

        const asm: {
            text: string;
            source: AsmResultSource | null;
            labels: InlineLabel[];
        }[] = [];
        let labelDefinitions: [string, number][] = [];

        let asmLines: string[] = this.cleanAsm(utils.splitLines(asmResult));
        let result = this.scanLabelsAndMethods(asmLines, filters.labels!);
        let sourceByLine = this.computeSourceMappingForAsmLines(asmLines, result);
        const startingLineCount = asmLines.length;

        if (filters.commentOnly) {
            const commentRe = /^\s*(;.*)$/;
            const filteredLines: string[] = [];
            const filteredSources: Array<AsmResultSource | null> = [];
            const lineMap: Record<number, number> = {};

            for (const index in asmLines) {
                if (!commentRe.test(asmLines[index])) {
                    lineMap[index] = filteredLines.length;
                    filteredLines.push(asmLines[index]);
                    filteredSources.push(sourceByLine[index]);
                }
            }

            asmLines = filteredLines;
            sourceByLine = filteredSources;

            const remappedResult: ScanLabelsAndMethodsResult = {
                labelDef: {},
                labelUsage: {},
                methodDef: {},
                methodUsage: {},
                allAvailable: result.allAvailable,
            };
            for (const line in lineMap) {
                const originalLine = Number.parseInt(line, 10);
                const remappedLine = lineMap[originalLine];
                if (result.labelDef[originalLine] !== undefined) {
                    remappedResult.labelDef[remappedLine] = result.labelDef[originalLine];
                }
                if (result.labelUsage[originalLine] !== undefined) {
                    remappedResult.labelUsage[remappedLine] = result.labelUsage[originalLine];
                }
                if (result.methodDef[originalLine] !== undefined) {
                    remappedResult.methodDef[remappedLine] = result.methodDef[originalLine];
                }
                if (result.methodUsage[originalLine] !== undefined) {
                    remappedResult.methodUsage[remappedLine] = result.methodUsage[originalLine];
                }
            }
            result = remappedResult;
        }

        for (const i in result.labelDef) {
            const label = result.labelDef[i];
            labelDefinitions.push([label.name, Number.parseInt(i, 10)]);
        }

        for (const i in result.methodDef) {
            const method = result.methodDef[i];
            labelDefinitions.push([method, Number.parseInt(i, 10)]);
        }

        for (const line in asmLines) {
            if (result.labelDef[line]?.remove) continue;

            const labels: InlineLabel[] = [];
            const label = result.labelUsage[line] || result.methodUsage[line];
            if (label && result.allAvailable.has(label.name)) {
                labels.push(label);
            }

            asm.push({
                text: asmLines[line],
                source: sourceByLine[line],
                labels,
            });
        }

        let lineOffset = 1;
        labelDefinitions = labelDefinitions.sort((a, b) => a[1] - b[1]);

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
            asm,
            labelDefinitions: Object.fromEntries(labelDefinitions.filter(i => i[1] !== -1)),
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
