// Copyright (c) 2018, Compiler Explorer Authors
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

import type {IRResultLine} from '../types/asmresult/asmresult.interfaces.js';

import * as utils from './utils.js';
import {LLVMIrBackendOptions} from '../types/compilation/ir.interfaces.js';
import {LLVMIRDemangler} from './demangler/llvm.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {isString} from '../shared/common-utils.js';

type MetaNode = {
    metaId: string;
    metaType: string;
};

export class LlvmIrParser {
    private maxIrLines: number;
    private debugReference: RegExp;
    private metaNodeRe: RegExp;
    private otherMetaDirective: RegExp;
    private namedMetaDirective: RegExp;
    private metaNodeOptionsRe: RegExp;
    private llvmDebugLine: RegExp;
    private llvmDebugAnnotation: RegExp;
    private otherMetadataAnnotation: RegExp;
    private attributeAnnotation: RegExp;
    private attributeDirective: RegExp;
    private moduleMetadata: RegExp;
    private functionAttrs: RegExp;
    private commentOnly: RegExp;
    private commentAtEOL: RegExp;

    constructor(
        compilerProps,
        private readonly irDemangler: LLVMIRDemangler,
    ) {
        this.maxIrLines = 5000;
        if (compilerProps) {
            this.maxIrLines = compilerProps('maxLinesOfAsm', this.maxIrLines);
        }

        this.debugReference = /!dbg (!\d+)/;
        this.metaNodeRe = /^(!\d+) = (?:distinct )?!DI([A-Za-z]+)\(([^)]+?)\)/;
        this.otherMetaDirective = /^(!\d+) = (?:distinct )?!{.*}/;
        this.namedMetaDirective = /^(![.A-Z_a-z-]+) = (?:distinct )?!{.*}/;
        this.metaNodeOptionsRe = /(\w+): (!?\d+|\w+|""|"(?:[^"]|\\")*[^\\]")/gi;

        this.llvmDebugLine = /^\s*call void @llvm\.dbg\..*$/;
        this.llvmDebugAnnotation = /,? !dbg !\d+/;
        this.otherMetadataAnnotation = /,? !(?!dbg)[\w.]+ (!\d+)/;
        this.attributeAnnotation = /,? #\d+(?= )/;
        this.attributeDirective = /^attributes #\d+ = { .+ }$/;
        this.functionAttrs = /^; Function Attrs: .+$/;
        this.moduleMetadata = /^((source_filename|target datalayout|target triple) = ".+"|; ModuleID = '.+')$/;
        this.commentOnly = /^\s*;.*$/;
        this.commentAtEOL = /\s*;.*$/;
    }

    getFileName(debugInfo, scope): string | null {
        const stdInLooking = /.*<stdin>|^-$|example\.[^/]+$|<source>/;

        if (!debugInfo[scope]) {
            // No such meta info.
            return null;
        }
        // MetaInfo is a file node
        if (debugInfo[scope].filename) {
            const filename = debugInfo[scope].filename;
            return stdInLooking.test(filename) ? null : filename;
        }
        // MetaInfo has a file reference.
        if (debugInfo[scope].file) {
            return this.getFileName(debugInfo, debugInfo[scope].file);
        }
        if (!debugInfo[scope].scope) {
            // No higher scope => can't find file.
            return null;
        }
        // "Bubbling" up.
        return this.getFileName(debugInfo, debugInfo[scope].scope);
    }

    getSourceLineNumber(debugInfo, scope) {
        if (!debugInfo[scope]) {
            return null;
        }
        if (debugInfo[scope].line) {
            return Number(debugInfo[scope].line);
        }
        if (debugInfo[scope].scope) {
            return this.getSourceLineNumber(debugInfo, debugInfo[scope].scope);
        }
        return null;
    }

    getSourceColumn(debugInfo, scope): number | undefined {
        if (!debugInfo[scope]) {
            return;
        }
        if (debugInfo[scope].column) {
            return Number(debugInfo[scope].column);
        }
        if (debugInfo[scope].scope) {
            return this.getSourceColumn(debugInfo, debugInfo[scope].scope);
        }
    }

    parseMetaNode(line: string) {
        // Metadata Nodes
        // See: https://llvm.org/docs/LangRef.html#metadata
        const match = line.match(this.metaNodeRe);
        if (!match) {
            return null;
        }
        const metaNode = {
            metaId: match[1],
            metaType: match[2],
        };

        let keyValuePair;
        while ((keyValuePair = this.metaNodeOptionsRe.exec(match[3]))) {
            const key = keyValuePair[1];
            metaNode[key] = keyValuePair[2];
            // Remove "" from string
            if (metaNode[key][0] === '"') {
                metaNode[key] = metaNode[key].substr(1, metaNode[key].length - 2);
            }
        }

        return metaNode;
    }

    async processIr(ir: string, options: LLVMIrBackendOptions) {
        const result: IRResultLine[] = [];
        const irLines = utils.splitLines(ir);
        const debugInfo: Record<string, MetaNode> = {};
        // Set to true initially to prevent any leading newlines as a result of filtering
        let prevLineEmpty = true;

        const filters: RegExp[] = [];
        const lineFilters: RegExp[] = [];

        if (options.filterDebugInfo) {
            filters.push(this.llvmDebugLine);
            lineFilters.push(this.llvmDebugAnnotation);
        }
        if (options.filterIRMetadata) {
            filters.push(this.moduleMetadata);
            filters.push(this.metaNodeRe);
            filters.push(this.otherMetaDirective);
            filters.push(this.namedMetaDirective);
            lineFilters.push(this.otherMetadataAnnotation);
        }
        if (options.filterAttributes) {
            filters.push(this.attributeDirective);
            filters.push(this.functionAttrs);
            lineFilters.push(this.attributeAnnotation);
        }
        if (options.filterComments) {
            filters.push(this.commentOnly);
            lineFilters.push(this.commentAtEOL);
        }

        for (const line of irLines) {
            if (line.trim().length === 0) {
                // Avoid multiple successive empty lines.
                if (!prevLineEmpty) {
                    result.push({text: ''});
                }
                prevLineEmpty = true;
            } else {
                let newLine = line;
                // eslint-disable-next-line no-constant-condition
                while (true) {
                    const temp = newLine;
                    for (const re of lineFilters) {
                        newLine = newLine.replace(re, '');
                    }
                    if (newLine === temp) {
                        break;
                    }
                }

                const resultLine: IRResultLine = {
                    text: newLine,
                };

                // Non-Meta IR line. Metadata is attached to it using "!dbg !123"
                const debugReferenceMatch = line.match(this.debugReference);
                if (debugReferenceMatch) {
                    resultLine.scope = debugReferenceMatch[1];
                }

                const metaNode = this.parseMetaNode(line);
                if (metaNode) {
                    debugInfo[metaNode.metaId] = metaNode;
                }

                // Filtering a full line
                if (filters.some(re => line.match(re))) {
                    continue;
                }

                result.push(resultLine);
                prevLineEmpty = false;
            }
        }

        if (result.length >= this.maxIrLines) {
            result.length = this.maxIrLines + 1;
            result[this.maxIrLines] = {text: '[truncated; too many lines]'};
        }

        for (const line of result) {
            if (!line.scope) continue;
            line.source = {
                file: this.getFileName(debugInfo, line.scope),
                line: this.getSourceLineNumber(debugInfo, line.scope),
                column: this.getSourceColumn(debugInfo, line.scope),
            };
        }

        if (options.demangle && this.irDemangler.canDemangle()) {
            return {
                asm: (await this.irDemangler.process({asm: result})).asm,
                languageId: 'llvm-ir',
            };
        } else {
            return {
                asm: result,
                languageId: 'llvm-ir',
            };
        }
    }

    async processFromFilters(ir, filters: ParseFiltersAndOutputOptions) {
        if (isString(ir)) {
            return await this.processIr(ir, {
                filterDebugInfo: !!filters.debugCalls,
                filterIRMetadata: !!filters.directives,
                filterAttributes: false,
                filterComments: !!filters.commentOnly,
                demangle: !!filters.demangle,
                // discard value names is handled earlier
            });
        }
        return {
            asm: [],
        };
    }

    async process(ir: string, irOptions: LLVMIrBackendOptions) {
        return await this.processIr(ir, irOptions);
    }

    isLlvmIr(code) {
        return code.includes('@llvm') && code.includes('!DI') && code.includes('!dbg');
    }
}
