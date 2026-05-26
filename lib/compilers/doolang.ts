import { BaseCompiler } from '../base-compiler.js';

export class DoolangCompiler extends BaseCompiler {
    static get key() {
        return 'doolang';
    }

    override optionsForFilter(filters, outputFilename, userOptions) {
        return ['build', '--keep-ll'].concat(userOptions || []);
    }

    override getOutputFilename(dirPath, outputFilebase) {
        return dirPath + '/output.ll';
    }
}
