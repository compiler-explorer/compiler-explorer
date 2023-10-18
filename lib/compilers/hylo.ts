import {BaseCompiler} from '../base-compiler.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {CompileChildLibraries} from '../../types/compilation/compilation.interfaces.js';

export class HyloCompiler extends BaseCompiler {
    static get key() {
        return 'hylo';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        // TODO: support LLVM IR view.
        // this.compiler.supportsIrView = true;
    }

    override getSharedLibraryPathsAsArguments(
        libraries: CompileChildLibraries[],
        libDownloadPath?: string,
        toolchainPath?: string,
    ) {
        return [];
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            return ['--emit', 'intel-asm', '-o', this.filename(outputFilename)];
        } else {
            return ['-o', this.filename(outputFilename)];
        }
    }
}
