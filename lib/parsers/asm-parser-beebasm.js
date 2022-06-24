import * as utils from '../utils';

import {AsmParser} from './asm-parser';

export class AsmParserBeebAsm extends AsmParser {
    constructor(compilerProps) {
        super(compilerProps);

        this.labelDef = /^(\.\w+)/i;
        this.asmOpcodeRe = /^\s*(?<address>[\dA-F]+)\s*(?<opcodes>([\dA-F]{2} ?)+)\s*(?<disasm>.*)/;
    }

    processAsm(asm, filters) {
        const startTime = process.hrtime.bigint();

        const asmLines = [];
        const labelDefinitions = {};

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
                const opcodes = (addressAndInstructionMatch.groups.opcodes || '').split(' ').filter(x => !!x);
                const address = parseInt(addressAndInstructionMatch.groups.address, 16);
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
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
