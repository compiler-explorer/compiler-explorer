import path from 'path';
import {BaseCompiler} from '../base-compiler.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

export class PythranCompiler extends BaseCompiler {
    static get key() {
        return 'pythran';
    }

    cpp_compiler_root: string;

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.cpp_compiler_root = this.compilerProps<string>(`compiler.${this.compiler.id}.cpp_compiler_root`);
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.cpp_compiler_root) {
            execOptions.env.PATH = path.join(this.cpp_compiler_root, 'bin');
            const ld_library_path = [
                path.join(this.cpp_compiler_root, 'lib'),
                path.join(this.cpp_compiler_root, 'lib64'),
            ];
            execOptions.env.LD_LIBRARY_PATH = ld_library_path.join(':');
        }

        return execOptions;
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        let options = ['-o', this.filename(outputFilename)];
        if (!filters.binary && !filters.binaryObject) options = options.concat('-E');
        return options;
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        if (typeof filters !== 'undefined' && filters.binary) return 'asm';
        else return 'cppp';
    }
}
