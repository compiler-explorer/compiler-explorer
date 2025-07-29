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

import {AsmResultSource, ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {PropertyGetter} from '../properties.interfaces.js';
import * as utils from '../utils.js';

import {AsmParser} from './asm-parser.js';

export class PTXAsmParser extends AsmParser {
    protected commentOnlyLine: RegExp;
    protected emptyLine: RegExp;
    protected override directive: RegExp;
    protected externFuncDirective: RegExp;
    protected paramDirective: RegExp;
    protected visibleDirective: RegExp;
    protected override fileFind: RegExp;
    protected override sourceTag: RegExp;
    protected externFuncStart: RegExp;
    protected openParen: RegExp;
    protected closeParen: RegExp;
    protected semicolon: RegExp;
    protected ptxDataDeclaration: RegExp;
    protected functionStart: RegExp;
    protected functionEnd: RegExp;
    protected callInstruction: RegExp;
    protected functionCallParam: RegExp;
    protected functionCallEnd: RegExp;
    protected labelLine: RegExp;

    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);

        this.commentOnlyLine = /^\s*\/\//;
        this.emptyLine = /^\s*$/;

        this.directive = /^\s*\..*$/;
        this.externFuncDirective = /^\s*\.extern\s+\.func/;
        this.paramDirective = /^\s*\.param/;
        this.visibleDirective = /^\s*\.visible/;

        this.fileFind = /^\s*\.file\s+(\d+)\s+"([^"]+)"/;
        this.sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)/;

        this.externFuncStart = /^\s*\.extern\s+\.func/;
        this.openParen = /^\s*\(\s*$/;
        this.closeParen = /^\s*\)\s*$/;
        this.semicolon = /^\s*;\s*$/;

        this.ptxDataDeclaration =
            /^\s*\.global\s+\.align\s+\d+\s+\.(b8|b16|b32|b64|u8|u16|u32|u64|s8|s16|s32|s64|f16|f32|f64)/;

        this.functionStart = /^\s*\{/;
        this.functionEnd = /^\s*\}/;

        this.callInstruction = /^\s*call\./;
        this.functionCallParam = /^\s*[a-zA-Z_][a-zA-Z0-9_]*,?\s*$/;
        this.functionCallEnd = /^\s*\)\s*;\s*$/;

        this.labelLine = /^\s*\$?[a-zA-Z_][a-zA-Z0-9_]*:.*$/;
    }

    override processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();

        const asm: ParsedAsmResultLine[] = [];
        const asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;

        const files = this.parseFiles(asmLines);

        let currentSource: AsmResultSource | null = null;

        let inExternFuncDeclaration = false;
        let externFuncSeenCloseParen = false;
        let inFunctionImplementation = false;
        let functionBraceDepth = 0;
        let inFunctionCall = false;
        let callBaseIndent = '';
        let braceDepth = 0;

        for (let line of asmLines) {
            const newSource = this.processSourceLine(line, files);
            if (newSource) {
                currentSource = newSource;
            }

            if (this.externFuncStart.test(line)) {
                inExternFuncDeclaration = true;
                externFuncSeenCloseParen = false;
            } else if (inExternFuncDeclaration && this.closeParen.test(line)) {
                externFuncSeenCloseParen = true;
            }

            if (this.functionStart.test(line)) {
                functionBraceDepth++;
                if (functionBraceDepth === 1) {
                    inFunctionImplementation = true;
                }
            } else if (this.functionEnd.test(line)) {
                functionBraceDepth--;
                if (functionBraceDepth === 0) {
                    inFunctionImplementation = false;
                }
            }

            if (this.callInstruction.test(line)) {
                inFunctionCall = true;
                const match = line.match(/^(\s*)/);
                callBaseIndent = match ? match[1] + '\t' : '\t';
            } else if (inFunctionCall && this.functionCallEnd.test(line)) {
                inFunctionCall = false;
                callBaseIndent = '';
            }

            if (filters.libraryCode && inExternFuncDeclaration) {
                if (externFuncSeenCloseParen && this.semicolon.test(line)) {
                    inExternFuncDeclaration = false;
                    externFuncSeenCloseParen = false;
                }
                continue;
            }

            if (filters.commentOnly) {
                if (this.commentOnlyLine.test(line) || this.emptyLine.test(line)) {
                    continue;
                }

                const commentIndex = line.indexOf('//');
                if (commentIndex > 0) {
                    line = line.substring(0, commentIndex).trimEnd();
                }
            }

            if (filters.directives && this.shouldSkipPTXDirective(line, inFunctionImplementation)) {
                continue;
            }

            let processedLine = line;

            // Never indent labels
            if (this.labelLine.test(line)) {
                processedLine = line.trim();
            } else {
                let indentLevel = braceDepth;

                // Adjust indent for closing braces - they should be at the same level as opening brace
                if (this.functionEnd.test(line)) {
                    indentLevel = Math.max(0, braceDepth - 1);
                }

                if (indentLevel > 0) {
                    const additionalIndent = '\t'.repeat(indentLevel);
                    processedLine = additionalIndent + line.trim();
                } else if (inFunctionCall && !this.functionCallEnd.test(line)) {
                    processedLine = this.improveCallIndentation(line, callBaseIndent);
                }
            }

            if (this.functionStart.test(line)) {
                braceDepth++;
            } else if (this.functionEnd.test(line)) {
                braceDepth--;
            }

            if (filters.trim) {
                processedLine = this.applyTrimFilter(processedLine);
            }

            asm.push({
                text: processedLine,
                source: this.hasOpcode(line) ? currentSource : null,
                labels: [],
            });
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: {},
            languageId: 'ptx',
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }

    private shouldSkipPTXDirective(line: string, inFunctionImplementation: boolean): boolean {
        if (!this.directive.test(line)) {
            return false;
        }

        if (this.externFuncDirective.test(line)) {
            return false;
        }

        if (this.paramDirective.test(line)) {
            return inFunctionImplementation;
        }

        if (this.visibleDirective.test(line)) {
            return false;
        }

        if (this.ptxDataDeclaration.test(line)) {
            return false;
        }

        return true;
    }

    private processSourceLine(line: string, files: Record<number, string>): AsmResultSource | null {
        const locMatch = line.match(this.sourceTag);
        if (locMatch) {
            const fileNum = Number.parseInt(locMatch[1]);
            const lineNum = Number.parseInt(locMatch[2]);
            const columnNum = Number.parseInt(locMatch[3]);

            const file = files[fileNum];
            if (file) {
                return {
                    file: utils.maskRootdir(file),
                    line: lineNum,
                    column: columnNum,
                    mainsource: true,
                };
            }
        }

        return null;
    }

    private improveCallIndentation(line: string, baseIndent: string): string {
        const trimmed = line.trim();
        if (!trimmed) {
            return line;
        }

        const currentIndent = line.match(/^(\s*)/)?.[1] || '';

        if (/^[a-zA-Z_][a-zA-Z0-9_]*,?\s*$/.test(trimmed)) {
            return currentIndent + baseIndent.slice(currentIndent.length) + trimmed;
        }

        if (trimmed === '(') {
            return currentIndent + baseIndent.slice(currentIndent.length) + trimmed;
        }

        if (trimmed === ')' || trimmed === ');') {
            const adjustedIndent = baseIndent.slice(0, -1);
            return currentIndent + adjustedIndent.slice(currentIndent.length) + trimmed;
        }

        return line;
    }

    private applyTrimFilter(line: string): string {
        if (line.trim().length === 0) {
            return '';
        }

        // Convert tabs to 2 spaces while preserving the indentation structure
        const leadingTabsMatch = line.match(/^(\t*)/);
        const leadingTabs = leadingTabsMatch ? leadingTabsMatch[1].length : 0;
        const leadingSpaces = '  '.repeat(leadingTabs);

        const contentAfterTabs = line.substring(leadingTabs);
        const contentWithSpaces = contentAfterTabs.replace(/\t/g, ' ');

        // Squash multiple spaces in the content part but preserve the leading indentation
        const contentParts = contentWithSpaces.split(/(\s+)/);
        const processedContent = contentParts
            .map((part, index) => {
                if (index === 0) return part;
                if (/^\s+$/.test(part)) return ' ';
                return part;
            })
            .join('');

        return leadingSpaces + processedContent;
    }
}
