import {BaseCompiler} from '../base-compiler.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

export class TableGenCompiler extends BaseCompiler {
    static get key() {
        return 'tablegen';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options: string[] = ['-o', outputFilename];
        if (this.compiler.includePath) {
            options.push(`-I${this.compiler.includePath}`);
        }
        return options;
    }

    override isCfgCompiler() {
        return false;
    }
}
