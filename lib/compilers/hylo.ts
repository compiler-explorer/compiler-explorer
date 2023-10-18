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
        let options = ['-o', this.filename(outputFilename)];
        // Theres's no equivalent to non-intel asm.
        if (!filters.binary && !filters.binaryObject) options = options.concat('--emit', 'intel-asm');
        return options;
    }
}
