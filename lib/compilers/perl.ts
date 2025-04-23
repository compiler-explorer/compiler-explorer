import {BaseCompiler} from '../base-compiler.js';

export class PerlCompiler extends BaseCompiler {
    static get key() {
        return 'perl';
    }

    override optionsForFilter(filters: Record<string, boolean>, outputFilename: string) {
        return ['-c'];
    }
}
