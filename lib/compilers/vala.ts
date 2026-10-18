// Copyright (c) 2023, Compiler Explorer Authors
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

import path from 'node:path';

import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {AsmParserCpp} from '../parsers/asm-parser-cpp.js';

export class ValaCompiler extends BaseCompiler {
    static get key() {
        return 'vala';
    }

    ccPath: string;
    pkgconfigPath: string;
    generatedCodeParser = new AsmParserCpp();

    constructor(compiler: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compiler, env);
        this.ccPath = this.compilerProps<string>(`compiler.${this.compiler.id}.cc`);
        this.pkgconfigPath = this.compilerProps<string>(`compiler.${this.compiler.id}.pkgconfigpath`);
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'c';
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        // the binary output is a disassembly; otherwise it is the generated C, which the asm parser would mangle
        if (filters.binary) return super.processAsm(result, filters, options);
        return this.generatedCodeParser.process(result.asm, filters);
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.ccPath) {
            execOptions.env.CC = this.ccPath;
        }

        if (this.pkgconfigPath) {
            execOptions.env.PKG_CONFIG_PATH = this.pkgconfigPath;
        }

        return execOptions;
    }

    override async populatePossibleOverrides() {
        this.addCCompilerOverride(false);
        await super.populatePossibleOverrides();
    }

    override applyOverridesToExecOptions(execOptions: ExecutionOptions, overrides: ConfiguredOverrides): void {
        super.applyOverridesToExecOptions(execOptions, overrides);
        const cc = this.getOverriddenCCompiler(overrides);
        if (cc && execOptions.env) execOptions.env.CC = cc;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: any) {
        const options = ['-g'];
        if (filters.binary) {
            options.push('-o', this.filename(outputFilename));
        } else {
            options.push('-C'); // Save transpiled C code
        }

        return options;
    }

    override async objdump(
        outputFilename: string,
        result: any,
        maxSize: number,
        intelAsm: boolean,
        demangle: boolean,
        staticReloc: boolean,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const objdumpResult = await super.objdump(
            outputFilename,
            result,
            maxSize,
            intelAsm,
            demangle,
            staticReloc,
            dynamicReloc,
            filters,
        );

        objdumpResult.languageId = 'asm';
        return objdumpResult;
    }

    override getSharedLibraryPathsAsArguments(
        libraries: SelectedLibraryVersion[],
        libDownloadPath: string | undefined,
    ) {
        return [];
    }

    override getSharedLibraryLinks(libraries: SelectedLibraryVersion[]): string[] {
        return [];
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, 'example.c');
    }
}
