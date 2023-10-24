import {BaseCompiler} from '../base-compiler.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

export class TableGenCompiler extends BaseCompiler {
    static get key() {
        return 'tablegen';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['-o', outputFilename];
    }

    override isCfgCompiler() {
        return false;
    }
}
