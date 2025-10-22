import path from 'node:path';
import {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {IAsmParser} from '../parsers/asm-parser.interfaces.js';
import {AsmRegex} from '../parsers/asmregex.js';
import * as utils from '../utils.js';
import {C2RustParser} from './argument-parsers.js';

// Simple parser that only does bare minimum, handling whitespace. In particular, Rust
// comments `#[]` etc are important, and as this isn't even slightly "assembly" we don't
// try and do anything like "dead code removal".
class NoOpParser implements IAsmParser {
    process(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        let asmLines = utils.splitLines(asmResult);
        if (filters.preProcessLines !== undefined) {
            asmLines = filters.preProcessLines(asmLines);
        }
        const asm = asmLines.map(line => {
            line = utils.expandTabs(line);

            const text = AsmRegex.filterAsmLine(line, filters);

            return {text, source: null, labels: []};
        });

        return {asm};
    }
}

export class C2RustCompiler extends BaseCompiler {
    static get key() {
        return 'c2rust';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.asm = new NoOpParser();
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

    override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ): string[] {
        return options.concat(
            [this.filename(inputFilename)],
            userOptions,
            userOptions.includes('--') ? [] : ['--'],
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            staticLibLinks,
        );
    }
}
