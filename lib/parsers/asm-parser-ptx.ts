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

            if (filters.directives && this.shouldSkipPTXDirective(line)) {
                continue;
            }

            asm.push({
                text: line,
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

    private shouldSkipPTXDirective(line: string): boolean {
        if (!this.directive.test(line)) {
            return false;
        }

        if (this.externFuncDirective.test(line)) {
            return false;
        }

        if (this.paramDirective.test(line)) {
            return false;
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
}
