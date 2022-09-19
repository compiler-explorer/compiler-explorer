import path from 'path';

import {BaseCompiler} from '../base-compiler';

export class HookCompiler extends BaseCompiler {
    static get key() {
        return 'hook';
    }

    optionsForFilter(filters) {
        return ['--dump'];
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, 'example.out');
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        const dirPath = path.dirname(inputFilename);
        const outputFilename = this.getOutputFilename(dirPath);
        options.push(outputFilename);
        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }
}
