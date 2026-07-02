// Copyright (c) 2023, Compiler Explorer Authors
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
import {PropertyGetter} from '../properties.interfaces.js';
import * as utils from '../utils.js';
import {AsmParser} from './asm-parser.js';

export class AsmParserBeebAsm extends AsmParser {
    constructor(compilerProps: PropertyGetter) {
        super(compilerProps);

        this.labelDef = /^(\.\w+)/i;
        this.asmOpcodeRe = /^\s*(?<address>[\dA-F]+)\s*(?<opcodes>([\dA-F]{2} ?)+)\s*(?<disasm>.*)/;
    }

    override processAsm(asm: string, _filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();

        const asmLines: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        let startingLineCount = 0;

        utils.eachLine(asm, line => {
            startingLineCount++;

            const labelMatch = line.match(this.labelDef);
            if (labelMatch) {
                asmLines.push({
                    text: line,
                });
                labelDefinitions[labelMatch[1]] = asmLines.length;
                return;
            }

            const addressAndInstructionMatch = line.match(this.asmOpcodeRe);
            if (addressAndInstructionMatch) {
                assert(addressAndInstructionMatch.groups);
                const opcodes = (addressAndInstructionMatch.groups.opcodes || '').split(' ').filter(x => !!x);
                const address = Number.parseInt(addressAndInstructionMatch.groups.address, 16);
                asmLines.push({
                    address: address,
                    opcodes: opcodes,
                    text: '  ' + addressAndInstructionMatch.groups.disasm,
                });
            }
        });

        const endTime = process.hrtime.bigint();

        return {
            asm: asmLines,
            labelDefinitions: labelDefinitions,
            parsingTime: utils.deltaTimeNanoToMili(startTime, endTime),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
