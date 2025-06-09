import type {AsmResultSource, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {resolvePathFromAppRoot} from '../utils.js';

export class RakuCompiler extends BaseCompiler {
    private readonly disasmScriptPath: string;
    private readonly exepath: string;

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
        this.disasmScriptPath = resolvePathFromAppRoot('etc', 'scripts', 'disasms', 'moarvm_disasm.raku');
        this.exepath = this.compiler.exe;
    }

    static get key() {
        return 'raku';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return [this.disasmScriptPath, this.exepath, outputFilename];
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        const lineRe = /^ {5}annotation: ([^:]*):(\d+)$/;
        const frameHeadRe = /^ {2}Frame_(\d+) :$/;

        const bytecodeLines = result.asm.split('\n');

        const bytecodeResult: ParsedAsmResultLine[] = [];
        let lastLineNo: number | null = null;
        let sourceLoc: AsmResultSource | null = null;

        for (const line of bytecodeLines) {
            const matchLine = line.match(lineRe);
            const matchFrame = line.match(frameHeadRe);

            if (matchFrame) {
                lastLineNo = null;
                sourceLoc = {line: null, file: null};
            } else if (matchLine) {
                const lineno = Number.parseInt(matchLine[2]);
                sourceLoc = {line: lineno, file: null};
                lastLineNo = lineno;
            } else if (line) {
                sourceLoc = {line: lastLineNo, file: null};
            } else {
                sourceLoc = {line: null, file: null};
                lastLineNo = null;
            }

            bytecodeResult.push({text: line, source: sourceLoc});
        }

        return {asm: bytecodeResult};
    }
}
