import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

export class HyloCompiler extends BaseCompiler {
    static get key() {
        return 'hylo';
    }

    override getSharedLibraryPathsAsArguments(
        libraries: SelectedLibraryVersion[],
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
