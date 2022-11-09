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

import * as utils from '../utils';

import {AsmParser} from './asm-parser';

export class SPIRVAsmParser extends AsmParser {
    parseOpString(asmLines) {
        const opString = /^\s*%(\d+)\s+=\s+OpString\s+"([^"]+)"$/;
        const files = {};
        for (const line of asmLines) {
            const match = line.match(opString);
            if (match) {
                const lineNum = parseInt(match[1]);
                files[lineNum] = match[2];
            }
        }
        return files;
    }

    override processAsm(asmResult, filters) {
        const startTime = process.hrtime.bigint();

        const asm: any = [];

        let asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;
        if (filters.preProcessLines !== undefined) {
            asmLines = filters.preProcessLines(asmLines);
        }
        const opStrings = this.parseOpString(asmLines);

        const sourceTag = /^\s*OpLine\s+%(\d+)\s+(\d+)\s+(\d*)$/;
        const endBlock = /OpFunctionEnd/;
        const comment = /;/;
        const opLine = /OpLine/;
        const opNoLine = /OpNoLine/;
        const opExtDbg = /OpExtInst\s+%void\s+%\d+\s+Debug/;
        let source: any = null;

        for (let line of asmLines) {
            const match = line.match(sourceTag);
            if (match) {
                source = {
                    file: utils.maskRootdir(opStrings[parseInt(match[1])]),
                    line: parseInt(match[2]),
                    mainsource: true,
                };
                const sourceCol = parseInt(match[3]);
                if (!isNaN(sourceCol) && sourceCol !== 0) {
                    source.column = sourceCol;
                }
            }

            if (endBlock.test(line) || opNoLine.test(line)) {
                source = null;
            }

            if (filters.commentOnly && comment.test(line)) {
                continue;
            }
            if (filters.directives) {
                if (opLine.test(line) || opExtDbg.test(line) || opNoLine.test(line)) {
                    continue;
                }
            }

            line = utils.expandTabs(line);
            asm.push({
                text: line,
                source: source,
                labels: [],
            });
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: new Map<string, number>(),
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
