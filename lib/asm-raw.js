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

import { AsmRegex } from './asmregex';

export class AsmRaw extends AsmRegex {
    processBinaryAsm(asm, filters) {
        let result = [];
        const asmLines = asm.split('\n');
        const asmOpcodeRe = /^\s*([\da-f]+):\s*(([\da-f]{2} ?)+)\s*(.*)/;
        const labelRe = /^([\da-f]+)\s+<([^>]+)>:$/;
        const destRe = /.*\s([\da-f]+)\s+<([^>]+)>$/;
        let source = null;

        if (asmLines.length === 1 && asmLines[0][0] === '<') {
            return [{text: asmLines[0], source: null}];
        }

        for (const line of asmLines) {
            let match = line.match(labelRe);
            if (match) {
                result.push({text: match[2] + ':', source: null});
                continue;
            } else {
                match = line.match(this.labelDef);
                if (match) {
                    result.push({text: match[1] + ':', source: null});
                    continue;
                }
            }

            match = line.match(asmOpcodeRe);
            if (match) {
                const address = parseInt(match[1], 16);
                const opcodes = match[2].split(' ').filter(function (x) {
                    return x;
                });
                const disassembly = ' ' + AsmRegex.filterAsmLine(match[4], filters);
                let links = null;
                const destMatch = line.match(destRe);
                if (destMatch) {
                    links = [{
                        offset: disassembly.indexOf(destMatch[1]),
                        length: destMatch[1].length,
                        to: parseInt(destMatch[1], 16),
                    }];
                }
                result.push({opcodes: opcodes, address: address, text: disassembly, source: source, links: links});
            }
        }

        return {
            asm: result,
        };
    }

    process(asm, filters) {
        return this.processBinaryAsm(asm, filters);
    }
}
