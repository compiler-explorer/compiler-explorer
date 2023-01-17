import {AsmParser} from './asm-parser';
import {AsmRegex} from './asmregex';

export class AsmParserZ88dk extends AsmParser {
    constructor(compilerProps) {
        super(compilerProps);

        this.asmOpcodeRe = /^\s*(?<disasm>.*)\s*;\[(?<address>[\da-f]+)]\s*(?<opcodes>([\da-f]{2} ?)+)/;
    }

    processBinaryAsm(asmResult, filters) {
        const startTime = process.hrtime.bigint();
        const asm = [];
        const labelDefinitions = {};

        let asmLines = asmResult.split('\n');
        const startingLineCount = asmLines.length;
        const source = null;

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
            const labelsInLine = [];

            if (asm.length >= this.maxAsmLines) {
                if (asm.length === this.maxAsmLines) {
                    asm.push({
                        text: '[truncated; too many lines]',
                        source: null,
                        labels: labelsInLine,
                    });
                }
                continue;
            }

            const match = line.match(this.asmOpcodeRe);
            if (match) {
                const address = parseInt(match.groups.address, 16);
                const opcodes = (match.groups.opcodes || '').split(' ').filter(x => !!x);
                const disassembly = ' ' + AsmRegex.filterAsmLine(match.groups.disasm.trimEnd(), filters);
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

        const endTime = process.hrtime.bigint();

        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
