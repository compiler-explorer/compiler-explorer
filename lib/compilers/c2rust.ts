import path from 'node:path';

import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

import {C2RustParser} from './argument-parsers.js';

export class C2RustCompiler extends BaseCompiler {
    static get key() {
        return 'c2rust';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['transpile'];
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'example.rs');
    }

    override getArgumentParserClass() {
        return C2RustParser;
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'rust';
    }

    // TODO: By default the comments filter is enable and seems to be stripping
    // out the first line of the Rust output, likely because it's seeing the `#`
    // for the attribute on the first line as a comment. This function is trying
    // to disable that by setting all the filters to false, but that doesn't
    // seem to disable the comments filter by default. There's probably a
    // different way to address this issue, e.g. fixing the output language such
    // that the filter doesn't see `#` as a comment.
    override getDefaultFilters(): ParseFiltersAndOutputOptions {
        return {
            binary: false,
            execute: false,
            demangle: false,
            intel: false,
            commentOnly: false,
            directives: false,
            labels: false,
            optOutput: false,
            libraryCode: false,
            trim: false,
            binaryObject: false,
            debugCalls: false,
        };
    }
}
