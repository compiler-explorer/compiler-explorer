import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class CodonCompiler extends BaseCompiler {
    static get key() {
        return 'codon';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        return ['build', '-llvm', '-o', this.filename(outputFilename)];
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'llvm-ir';
    }
}
