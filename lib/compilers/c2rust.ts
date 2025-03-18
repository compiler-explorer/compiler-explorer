import path from 'node:path';

import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

import {C2RustParser} from './argument-parsers.js';

// TODO: By default the comments filter is enable and seems to be stripping out
// the first line of the Rust output, likely because it's seeing the `#` for the
// attribute on the first line as a comment. I tried using `getDefaultFilters`
// to set all the filters to false, but that doesn't seem to disable the
// comments filter by default. There's probably a different way to address this
// issue, e.g. fixing the output language such that the filter doesn't see `#`
// as a comment, but it's already set to Rust so I'm not sure what else needs to
// be done there.
export class C2RustCompiler extends BaseCompiler {
    static get key() {
        return 'c2rust';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['transpile'];
    }

    override getArgumentParserClass() {
        return C2RustParser;
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'rust';
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'example.rs');
    }
}
