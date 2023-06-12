import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

export class AsmXCompiler  extends BaseCompiler {
    static get key(){
        return 'AsmX';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['asmx-cli', 'build', 'arm', outputFilename, outputFilename];
    }
}
