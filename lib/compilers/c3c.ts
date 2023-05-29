import path from 'path';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

export class C3Compiler extends BaseCompiler {
    static get key() {
        return 'c3c';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit-llvm'];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['compile-only', '-g', '-l', 'pthread', '--emit-asm'];
    }

    override getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        return this.filename(path.dirname(inputFilename) + '/output.ll');
    }
}
