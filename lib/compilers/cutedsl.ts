// Copyright (c) 2026, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import * as fs from 'node:fs/promises';
import Path from 'node:path';

import type {
    CompilationInfo,
    CompilationResult,
    ExecutionOptionsWithEnv,
} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import type {IAsmParser} from '../parsers/asm-parser.interfaces.js';
import {MlirAsmParser} from '../parsers/asm-parser-mlir.js';
import {PTXAsmParser} from '../parsers/asm-parser-ptx.js';
import {SassAsmParser} from '../parsers/asm-parser-sass.js';
import {resolvePathFromAppRoot} from '../utils.js';
import {BaseParser} from './argument-parsers.js';

export class CuteDslCompiler extends BaseCompiler {
    private compilerWrapperPath: string;
    private parserMap: Record<string, IAsmParser>;

    static get key() {
        return 'cutedsl';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper', '') || resolvePathFromAppRoot('etc', 'scripts', 'cutedsl_wrapper.py');

        const ptxAsmParser = new PTXAsmParser();
        const sassAsmParser = new SassAsmParser(this.compilerProps);
        const mlirAsmParser = new MlirAsmParser();
        this.asm = ptxAsmParser;
        this.compiler.supportsDeviceAsmView = true;
        this.parserMap = {
            '.ptx': ptxAsmParser,
            '.sass': sassAsmParser,
            '.mlir': mlirAsmParser,
        };
    }

    override getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const execOptions = super.getDefaultExecOptions();
        execOptions.env.CUTE_DSL_KEEP = execOptions.env.CUTE_DSL_KEEP || this.compilerProps('keep', 'ir,ptx,cubin');
        execOptions.env.CUTE_DSL_NO_CACHE = execOptions.env.CUTE_DSL_NO_CACHE || this.compilerProps('noCache', '1');
        return execOptions;
    }

    override getCompilerResultLanguageId(_filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'ptx';
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        return Path.join(dirPath, `${outputFilebase}.ptx`);
    }

    override optionsForFilter(_filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        return ['-I', this.compilerWrapperPath, '--output_file', outputFilename];
    }

    override getArgumentParserClass() {
        return BaseParser;
    }

    override async extractDeviceCode(
        result: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        compilationInfo: CompilationInfo,
    ) {
        const devices = {...result.devices};
        const {dirPath} = result;
        if (!dirPath) {
            return result;
        }

        const outputFilename = Path.basename(compilationInfo.outputFilename);
        const files = await fs.readdir(dirPath);
        await Promise.all(
            files.map(async filename => {
                if (filename === outputFilename) {
                    return;
                }

                const ext = Path.extname(filename);
                const parser = this.parserMap[ext];
                if (!parser) {
                    return;
                }

                const data = await fs.readFile(Path.join(dirPath, filename), 'utf8');
                Object.assign(devices, {[filename]: await parser.process(data, filters)});
            }),
        );
        result.devices = devices;
        return result;
    }

    override getDefaultFilters() {
        return {
            intel: false,
            commentOnly: false,
            directives: false,
            labels: false,
            optOutput: true,
            binary: false,
            execute: false,
            demangle: false,
            libraryCode: false,
            trim: false,
            binaryObject: false,
            debugCalls: false,
        };
    }
}
