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

import {
    AsmResultLabel,
    AsmResultSource,
    ParsedAsmResult,
    ParsedAsmResultLine,
} from '../../types/asmresult/asmresult.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {assert} from '../assert.js';
import * as utils from '../utils.js';

import {AsmParser} from './asm-parser.js';
import {AsmRegex} from './asmregex.js';

export class DartAsmParser extends AsmParser {
    constructor() {
        super();

        this.lineRe = /^(file:)?(\/[^:]+):(?<line>\d+).*/;
    }

    override processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions = {};
        const dontMaskFilenames = filters.dontMaskFilenames;

        let asmLines = asmResult.split('\n');
        const startingLineCount = asmLines.length;
        let source: AsmResultSource | null = null;
        let func: string | null = null;
        let mayRemovePreviousLabel = filters.libraryCode;

        function maybeRemovePreviousLabel() {
            if (mayRemovePreviousLabel && func) {
                const previousLabelStart = labelDefinitions[func];
                if (previousLabelStart) {
                    asm.splice(previousLabelStart - 1);
                }
            }
        }

        // Handle "error" documents.
        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return {
                asm: [{text: asmLines[0], source: null}],
            };
        }

        if (filters.preProcessBinaryAsmLines !== undefined) {
            asmLines = filters.preProcessBinaryAsmLines(asmLines);
        }

        for (const line of asmLines) {
            const labelsInLine: AsmResultLabel[] = [];

            if (asm.length >= this.maxAsmLines && !mayRemovePreviousLabel) {
                if (asm.length === this.maxAsmLines) {
                    asm.push({
                        text: '[truncated; too many lines]',
                        source: null,
                        labels: labelsInLine,
                    });
                }
                continue;
            }
            let match = line.match(this.lineRe);
            if (match) {
                assert(match.groups);
                if (dontMaskFilenames) {
                    source = {
                        file: utils.maskRootdir(match[1]),
                        line: parseInt(match.groups.line),
                        mainsource: true,
                    };
                } else {
                    source = {file: null, line: parseInt(match.groups.line), mainsource: true};
                }
                continue;
            }

            match = line.match(this.labelRe);
            if (match) {
                maybeRemovePreviousLabel();
                mayRemovePreviousLabel = filters.libraryCode;
                source = null;
                func = match[2];
                if (this.isUserFunction(func)) {
                    asm.push({
                        text: func + ':',
                        source: null,
                        labels: labelsInLine,
                    });
                    labelDefinitions[func] = asm.length;
                }
                continue;
            }

            if (func && line === `${func}():`) continue;

            if (!func || !this.isUserFunction(func)) continue;

            // note: normally the source.file will be null if it's code from example.ext
            //  but with filters.dontMaskFilenames it will be filled with the actual filename
            //  instead we can test source.mainsource in that situation
            const isMainsource = source && (source.file === null || source.mainsource);
            if (isMainsource) {
                mayRemovePreviousLabel = false;
            }

            match = line.match(this.asmOpcodeRe);
            if (match) {
                assert(match.groups);
                const address = parseInt(match.groups.address, 16);
                const opcodes = (match.groups.opcodes || '').split(' ').filter(x => !!x);
                const disassembly = ' ' + AsmRegex.filterAsmLine(match.groups.disasm, filters);
                const destMatch = line.match(this.destRe);
                if (destMatch) {
                    const labelName = destMatch[2];
                    const startCol = disassembly.indexOf(labelName) + 1;
                    labelsInLine.push({
                        name: labelName,
                        range: {
                            startCol: startCol,
                            endCol: startCol + labelName.length,
                        },
                    });
                }
                asm.push({
                    opcodes: opcodes,
                    address: address,
                    text: disassembly,
                    source: source,
                    labels: labelsInLine,
                });
            }
        }

        maybeRemovePreviousLabel();

        this.removeLabelsWithoutDefinition(asm, labelDefinitions);

        const endTime = process.hrtime.bigint();

        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
