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

import {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {assert} from '../assert.js';
import {deltaTimeNanoToMili, splitLines} from '../utils.js';
import {IAsmParser} from './asm-parser.interfaces.js';
import {AsmRegex} from './asmregex.js';

/**
 * The parser for PolkaVM assembly.
 *
 * @note
 * There are currently no source mappings from PolkaVM.
 */
export class PolkaVMAsmParser implements IAsmParser {
    /**
     * @example `    64: ra = 6, jump @42`
     */
    protected instructionRe = /^\s+(?<address>\d+):\s+(?<disasm>.*)$/;
    /**
     * @example `<deploy>:`
     */
    protected labelRe = /^<(?<label>[^\s\n]+)>:$/;
    /**
     * @example `// Stack size = 32768 bytes`
     */
    protected headerCommentRe = /^\/\//;
    /**
     * @example `      // This is a comment`
     */
    protected commentOnlyRe = /^\s*\/\//;
    /**
     * @example `      : @16 (gas: 5)`
     */
    protected jumpTargetRe = /^\s*: (?<targetLine>@\d+.*)$/;
    protected maxAsmLines = 5000;
    protected indentation = '        ';

    process(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();

        let asmLines = splitLines(asmResult);
        const originalLineCount = asmLines.length;

        if (filters.preProcessLines) {
            asmLines = filters.preProcessLines(asmLines);
        }

        const {asm, labelDefinitions} = this.processLines(asmLines, filters);

        const endTime = process.hrtime.bigint();

        return {
            asm,
            labelDefinitions,
            parsingTime: deltaTimeNanoToMili(startTime, endTime),
            filteredCount: originalLineCount - asm.length,
        };
    }

    processLines(
        asmLines: string[],
        filters: ParseFiltersAndOutputOptions,
    ): {
        asm: ParsedAsmResultLine[];
        labelDefinitions: Record<string, number>;
    } {
        const parsedAsm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        for (const line of asmLines) {
            if (parsedAsm.length >= this.maxAsmLines) {
                parsedAsm.push({
                    text: '[truncated; too many lines]',
                    source: null,
                });
                break;
            }

            let match: RegExpMatchArray | null = null;

            if ((match = line.match(this.instructionRe))) {
                this.parseInstruction(parsedAsm, match, filters);
            } else if ((match = line.match(this.jumpTargetRe))) {
                this.parseJumpTarget(parsedAsm, match, labelDefinitions);
            } else if ((match = line.match(this.labelRe))) {
                this.parseLabel(parsedAsm, match, labelDefinitions);
            } else if ((match = line.match(this.commentOnlyRe))) {
                this.parseComment(parsedAsm, match, line, filters);
            }
        }

        return {asm: parsedAsm, labelDefinitions};
    }

    private parseInstruction(
        parsedAsm: ParsedAsmResultLine[],
        match: RegExpMatchArray,
        filters: ParseFiltersAndOutputOptions,
    ): void {
        assert(match.groups);
        const address = parseInt(match.groups.address, 10);
        const disassembly = this.indentation + AsmRegex.filterAsmLine(match.groups.disasm, filters);
        parsedAsm.push({
            address,
            text: disassembly,
            source: null,
        });
    }

    private parseJumpTarget(
        parsedAsm: ParsedAsmResultLine[],
        match: RegExpMatchArray,
        filters: ParseFiltersAndOutputOptions,
    ): void {
        assert(match.groups);
        const targetLine = this.indentation + AsmRegex.filterAsmLine(match.groups.targetLine, filters);
        parsedAsm.push({
            text: targetLine,
            source: null,
        });
    }

    private parseLabel(
        parsedAsm: ParsedAsmResultLine[],
        match: RegExpMatchArray,
        labelDefinitions: Record<string, number>,
    ): void {
        assert(match.groups);
        const label = match.groups.label;
        parsedAsm.push({
            text: label + ':',
            source: null,
        });
        labelDefinitions[label] = parsedAsm.length;
    }

    private parseComment(
        parsedAsm: ParsedAsmResultLine[],
        match: RegExpMatchArray,
        line: string,
        filters: ParseFiltersAndOutputOptions,
    ): void {
        if (!filters.commentOnly) {
            const trimmedInput = match.input?.trim() ?? '';
            const comment = line.match(this.headerCommentRe) ? trimmedInput : this.indentation + trimmedInput;
            parsedAsm.push({
                text: comment,
                source: null,
            });
        }
    }
}
