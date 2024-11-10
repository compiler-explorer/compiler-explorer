import path from 'path';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class PythranCompiler extends BaseCompiler {
    static get key() {
        return 'pythran';
    }

    cpp_compiler_root: string;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.cpp_compiler_root = this.compilerProps<string>(`compiler.${this.compiler.id}.cpp_compiler_root`);
    }

    override getSharedLibraryPaths(libraries: SelectedLibraryVersion[], dirPath?: string): string[] {
        let ldpath = super.getSharedLibraryPaths(libraries, dirPath);
        if (this.cpp_compiler_root) {
            ldpath = ldpath.concat(
                path.join(this.cpp_compiler_root, 'lib'),
                path.join(this.cpp_compiler_root, 'lib64'),
            );
        }
        return ldpath;
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();

        if (this.cpp_compiler_root) {
            execOptions.env.PATH = execOptions.env.PATH + ':' + path.join(this.cpp_compiler_root, 'bin');
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
        if (filters !== undefined && filters.binary) return 'asm';
        else return 'cppp';
    }
}
