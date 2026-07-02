import fs from 'node:fs/promises';
import path from 'node:path';

import type {FiledataPair} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {BasicExecutionResult, UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import type {CompilationEnvironment} from '../compilation-env.js';
import {parseRustOutput} from '../utils.js';
import {RustCompiler} from './rust.js';

export class Co2ccCompiler extends BaseCompiler {
    static get key() {
        return 'co2cc';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compileFilename = 'example.c';
        this.compiler.supportsIntel = true;
        this.compiler.intelAsm = '-masm=intel';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options = ['-o', outputFilename];
        if (!filters.binary && !filters.binaryObject) {
            options.unshift('-S');
        }
        if (this.compiler.intelAsm && filters.intel && !filters.binary && !filters.binaryObject) {
            options.push(this.compiler.intelAsm);
        }
        return options;
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getCompilerResultLanguageId(): string | undefined {
        return 'asm';
    }
}

export class Co2RustcCompiler extends RustCompiler {
    static override get key() {
        return 'co2rustc';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compileFilename = 'example.co2';
    }

    override async writeAllFiles(dirPath: string, source: string, files: FiledataPair[]) {
        if (!source) throw new Error(`File ${this.compileFilename} has no content or file is missing`);

        const co2Filename = path.join(dirPath, this.compileFilename);
        await fs.writeFile(co2Filename, source);

        const rsFilename = path.join(dirPath, 'example.rs');
        await fs.writeFile(rsFilename, '#![language(co2)]\n');

        if (files && files.length > 0) {
            await this.writeMultipleFiles(files, dirPath);
        }

        return {
            inputFilename: rsFilename,
        };
    }

    override processExecutionResult(input: UnprocessedExecResult, inputFilename?: string): BasicExecutionResult {
        const dirPath = inputFilename ? path.dirname(inputFilename) : undefined;
        const co2Filename = dirPath ? path.join(dirPath, this.compileFilename) : undefined;
        return {
            ...input,
            stdout: parseRustOutput(input.stdout, co2Filename),
            stderr: parseRustOutput(input.stderr, co2Filename),
        };
    }
}
