import {BaseCompiler} from '../base-compiler.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

export class CflatCompiler extends BaseCompiler {
    static get key() {
        return 'cflat';
    }
    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['compile-only', '-g', '-l', 'pthread', '--emit-asm'];
    }
}
