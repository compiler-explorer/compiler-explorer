// Copyright (c) 2018, 2023, Elliot Saba & Compiler Explorer Authors
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

import path from 'path';

import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import * as utils from '../utils.js';

import {JuliaParser} from './argument-parsers.js';

export class JuliaCompiler extends BaseCompiler {
    public compilerWrapperPath: string;

    static get key() {
        return 'julia';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--format=llvm-module'];
        this.compiler.minIrArgs = ['--format=llvm-module'];
        this.compiler.demangler = '';
        this.demanglerClass = null;
        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper', '') ||
            utils.resolvePathFromAppRoot('etc', 'scripts', 'julia_wrapper.jl');
    }

    // No demangling for now
    override postProcessAsm(result) {
        return result;
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return [];
    }

    override getArgumentParserClass() {
        return JuliaParser;
    }

    override fixExecuteParametersForInterpreting(
        executeParameters: ExecutableExecutionOptions,
        outputFilename: string,
    ) {
        super.fixExecuteParametersForInterpreting(executeParameters, outputFilename);
        (executeParameters.args as string[]).unshift('--');
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const dirPath = path.dirname(inputFilename);

        if (!execOptions.customCwd) {
            execOptions.customCwd = dirPath;
        }

        const juliaOptions = [this.compilerWrapperPath, '--'];
        juliaOptions.push(
            ...options,
            options.includes('--format=llvm-module')
                ? this.getIrOutputFilename(inputFilename, filters)
                : this.getOutputFilename(dirPath, this.outputFilebase),
        );

        const execResult = await this.exec(compiler, juliaOptions, execOptions);
        return {
            compilationOptions: juliaOptions,
            ...this.transformToCompilationResult(execResult, inputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
        };
    }
}
