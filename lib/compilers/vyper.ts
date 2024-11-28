import path from 'path';

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
                const pushBytes = parseInt(opcode.slice(4), 10);
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
