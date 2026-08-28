// Copyright (c) 2024, Compiler Explorer Authors
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

import path from 'node:path';

import type {
    AsmResultSource,
    ParsedAsmResult,
    ParsedAsmResultLine,
} from '../../types/asmresult/asmresult.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

export class VyperCompiler extends BaseCompiler {
    static get key() {
        return 'vyper';
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'output.txt');
    }

    override optionsForFilter(): string[] {
        return ['-f', 'opcodes,source_map', '-o', 'output.txt'];
    }

    override isCfgCompiler() {
        // TODO: it is possible to extract CFG using rattle and pc_jump_map, might implement in the future
        return false;
    }

    override async processAsm(result): Promise<ParsedAsmResult> {
        if (result.code !== 0) {
            const stderrOutput = result.stderr.map(entry => entry.text).join('\n');
            return {asm: [{text: result.asm + '\n\n' + stderrOutput}]};
        }

        const [opcodesStr, sourceMapStr] = result.asm.split('\n');
        const sourceMapObj = JSON.parse(sourceMapStr);

        const segments: ParsedAsmResultLine[] = [];
        const opcodesArray = opcodesStr.split(' ');

        const pcPosMap = sourceMapObj['pc_pos_map'];

        let pc = 0;

        for (let i = 0; i < opcodesArray.length; i++) {
            const opcode = opcodesArray[i];
            let disassembly = opcode;

            if (opcode.startsWith('PUSH') && opcode !== 'PUSH0' && i < opcodesArray.length - 1) {
                disassembly += ` ${opcodesArray[i + 1]}`;
                i++;
            }

            const source: AsmResultSource | null =
                pc in pcPosMap
                    ? {
                          file: null,
                          line: pcPosMap[pc][0],
                          column: pcPosMap[pc][1],
                          mainsource: true,
                      }
                    : null;

            const asmResultLine: ParsedAsmResultLine = {
                text: disassembly,
                opcodes: [opcode],
                address: pc,
                disassembly,
                source,
            };

            segments.push(asmResultLine);

            if (opcode.startsWith('PUSH')) {
                const pushBytes = Number.parseInt(opcode.slice(4), 10);
                pc += 1 + pushBytes;
            } else {
                pc += 1;
            }
        }

        return {
            asm: segments,
        };
    }
}
