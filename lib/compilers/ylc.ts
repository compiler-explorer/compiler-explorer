import { ParseFiltersAndOutputOptions } from "../../types/features/filters.interfaces.js";
import { BaseCompiler } from "../base-compiler.js";
import { BaseParser } from "./argument-parsers.js";

export class YLCCompiler extends BaseCompiler {
    static get key() {
        return 'ylc';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = ['-o=' + this.filename(outputFilename)];
        return options;
    }

    override getArgumentParserClass() {
        return BaseParser;
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }
}