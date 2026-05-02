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

type InlineLabel = {name: string; range: {startCol: number; endCol: number}};

function formatDotNetMethodKey(method: DotNetMethodKey) {
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
            for (let index = 0; index < Number.parseInt(arity[1], 10); index++, genericParameterIndex++) {
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

    constructor(sourceMapping: DotNetSourceMapping = []) {
        this.sourceMapping = sourceMapping;
        this.sourceMappingByName = new Map(
            sourceMapping.map(methodSourceMapping => [
                formatDotNetMethodKey(methodSourceMapping.method),
                methodSourceMapping,
            ]),
        );
    }

    private getMethodSourceMapping(methodName: string) {
        const requestedMethod = parseJitDisasmMethodKey(methodName);
        if (!requestedMethod) {
            return undefined;
        }

        const exactMapping = this.sourceMappingByName.get(formatDotNetMethodKey(requestedMethod));
        if (exactMapping) {
            return exactMapping;
        }

        if (requestedMethod.typeArguments.length === 0 && requestedMethod.methodArguments.length === 0) {
            return undefined;
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

            const candidateParameters = candidate.parameters.map(parameter =>
                substituteMetadataGenericParameters(parameter, requestedMethod),
            );
            const candidateReturnType = substituteMetadataGenericParameters(candidate.returnType, requestedMethod);

            if (
                candidateParameters.every((parameter, index) => parameter === requestedMethod.parameters[index]) &&
                candidateReturnType === requestedMethod.returnType
            ) {
                return methodSourceMapping;
            }
        }

        return undefined;
    }

    private getSourceForOffset(methodSourceMapping: DotNetMethodSourceMapping | undefined, offset: number) {
        if (!methodSourceMapping) {
            return null;
        }

        if (methodSourceMapping.offsets[offset] !== undefined) {
            return methodSourceMapping.offsets[offset];
        }

        let bestOffset: number | undefined;
        for (const candidate of Object.keys(methodSourceMapping.offsets)
            .map(Number)
            .sort((a, b) => a - b)) {
            if (candidate > offset) {
                break;
            }
            bestOffset = candidate;
        }

        return bestOffset === undefined ? null : methodSourceMapping.offsets[bestOffset];
    }

    private computeSourceMappingForAsmLines(asmLines: string[]) {
        const sources: Array<AsmResultSource | null> = [];
        let currentMethodSourceMapping: DotNetMethodSourceMapping | undefined;
        let currentSource: AsmResultSource | null = null;
        const inlineRootOffsetRe = /INLRT\s+@\s*0x(?<offset>[0-9a-fA-F]+)/g;
        const result = this.scanLabelsAndMethods(asmLines, false);

        for (const line in asmLines) {
            if (result.methodDef[line]) {
                currentMethodSourceMapping = this.getMethodSourceMapping(result.methodDef[line]);
                currentSource = null;
            }

            let lineSource: AsmResultSource | null = null;
            if (result.labelDef[line]) {
                currentSource = null;
            }

            const inlineRootOffsetMatches = Array.from(asmLines[line].matchAll(inlineRootOffsetRe));
            const inlineRootOffsetMatch = inlineRootOffsetMatches.at(-1);
            if (inlineRootOffsetMatch?.groups) {
                lineSource = null;
                currentSource = this.getSourceForOffset(
                    currentMethodSourceMapping,
                    Number.parseInt(inlineRootOffsetMatch.groups.offset, 16),
                );
            }

            const trimmedAsmLine = asmLines[line].trimStart();
            if (!inlineRootOffsetMatch && trimmedAsmLine !== '' && !trimmedAsmLine.startsWith(';')) {
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

    process(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();

        const asm: {
            text: string;
            source: AsmResultSource | null;
            labels: InlineLabel[];
        }[] = [];
        let labelDefinitions: [string, number][] = [];

        let asmLines: string[] = this.cleanAsm(utils.splitLines(asmResult));
        let sourceByLine = this.computeSourceMappingForAsmLines(asmLines);
        const startingLineCount = asmLines.length;

        if (filters.commentOnly) {
            const commentRe = /^\s*(;.*)$/;
            const filteredLines: string[] = [];
            const filteredSources: Array<AsmResultSource | null> = [];

            for (const index in asmLines) {
                if (!commentRe.test(asmLines[index])) {
                    filteredLines.push(asmLines[index]);
                    filteredSources.push(sourceByLine[index]);
                }
            }

            asmLines = filteredLines;
            sourceByLine = filteredSources;
        }

        const result = this.scanLabelsAndMethods(asmLines, filters.labels!);

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
            if (label) {
                if (result.allAvailable.includes(label.name)) {
                    labels.push(label);
                }
            }

            asm.push({
                text: asmLines[line],
                source: sourceByLine[line],
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
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
