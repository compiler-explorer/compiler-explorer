import { ParseFiltersAndOutputOptions } from "../../types/features/filters.interfaces.js";
import { BaseCompiler } from "../base-compiler.js";

export class YLCCompiler extends BaseCompiler {
    static get key() {
        return 'ylc';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = ['-o=' + this.filename(outputFilename), "-asm-clr"];
        return options;
    }
}