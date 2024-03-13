import {PropertyGetter} from '../properties.interfaces';

import {AsmParser} from './asm-parser';
import {IAsmParser} from './asm-parser.interfaces';

export class AsmParserTricoreGNU extends AsmParser implements IAsmParser {
    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);
        this.lineRe = /^(\w:[^:]+):(?<line>\d+).*\s*/;
        this.labelRe = /^([\da-f]+)\s+<(.+)>:\s*$/;
        this.asmOpcodeRe = /^\s*(?<address>[\da-f]+):\s*(?<opcodes>([\da-f]{2} ?)+)\s*(?<disasm>.*)\s*/;
        this.relocationRe = /^\s*(?<address>[\da-f]+):\s*(?<relocname>(R_[\dA-Z_]+))\s*(?<relocdata>.*)\s*/;
    }
}
