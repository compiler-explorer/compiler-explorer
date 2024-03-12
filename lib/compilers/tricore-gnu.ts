import {BaseCompiler} from '../base-compiler';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {CompilationEnvironment} from '../compilation-env';
import {AsmParserTricoreGNU} from '../parsers/asm-parser-tricore-gnu';

export class TricoreGNUCompiler extends BaseCompiler {
    override asm: AsmParserTricoreGNU;
    static get key() {
        return 'tricore-gnu';
    }

    constructor(info: CompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.asm = new AsmParserTricoreGNU(this.compilerProps);
    }
}
