import path from 'path';

import {CompileChildLibraries} from '../../types/compilation/compilation.interfaces.js';

import {BaseCompiler} from '../base-compiler.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

export class PythranCompiler extends BaseCompiler {
    static get key() {
        return 'pythran';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.compiler.cpp_compiler_root = this.compilerProps<string>(`compiler.${this.compiler.id}.cpp_compiler_root`);
    }

    override getSharedLibraryPaths(libraries: CompileChildLibraries[], dirPath?: string): string[] {
        let ldpath = super.getSharedLibraryPaths(libraries, dirPath);
        ldpath = ldpath.concat(
            path.join(this.compiler.cpp_compiler_root, 'lib'),
            path.join(this.compiler.cpp_compiler_root, 'lib64'),
        );
        return ldpath;
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();

        if (this.compiler.cpp_compiler_root) {
            execOptions.env.PATH = execOptions.env.PATH + ':' + path.join(this.compiler.cpp_compiler_root, 'bin');
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
