import {BaseCompiler} from '../base-compiler.js';

// TODO: remove this comment
// For reference, the basic behaviour of BaseCompiler is:
// - make a random temporary folder
// - save example.extension to the new folder, the full path to this is the inputFilename
// - the outputFilename is determined by the getOutputFilename() method
// - execute the compiler.exe with the arguments from OptionsForFilter() and adding inputFilename
// - be aware that the language class is only instanced once, so storing state is not possible

export class ValCompiler extends BaseCompiler {
    static get key() {
        return 'val';
    }

    // TODO
    override optionsForFilter(filters, outputFilename) {
        return [];
    }
}
