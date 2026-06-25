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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import path from 'node:path';

import {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class LeanCompiler extends BaseCompiler {
    // Flags that will be forwarded to `lean` instead of `leanc`.
    private readonly leanFlags = new Set(['--profile', '--stats']);

    static get key() {
        return 'lean';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsLeanCView = true;
    }

    override isCfgCompiler() {
        return true;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return [`--c=${this.getLeanCOutputFilename(outputFilename)}`];
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    private async getLeanPrefix(compiler: string, execOptions: ExecutionOptionsWithEnv): Promise<string | null> {
        const prefixResult = await this.exec(compiler, ['--print-prefix'], execOptions);
        const prefix = prefixResult.stdout.trim();
        if (prefixResult.code === 0 && prefix) {
            return prefix;
        }

        return null;
    }

    private async getLeancPath(compiler: string, execOptions: ExecutionOptionsWithEnv): Promise<string> {
        const prefix = await this.getLeanPrefix(compiler, execOptions);
        if (prefix) return path.join(prefix, 'bin', 'leanc');

        return path.join(path.dirname(compiler), 'leanc');
    }

    private async doLeanBuildstep(
        command: string,
        args: string[],
        execOptions: ExecutionOptionsWithEnv,
        inputFilename: string,
    ) {
        const result = await this.exec(command, args, execOptions);
        const processedResult = this.processExecutionResult(result, result.filenameTransform(inputFilename));

        for (const output of [processedResult.stdout, processedResult.stderr]) {
            for (const line of output) {
                if (line.tag?.column !== undefined) {
                    // Lean reports zero-based columns, Monaco diagnostics use one-based columns
                    line.tag.column++;
                }
            }
        }

        return processedResult;
    }

    private partitionOptionsForSteps(
        options: string[],
        inputFilename: string,
    ): {
        leanOptions: string[];
        leancOptions: string[];
    } {
        const leanOptions: string[] = [];
        const leancOptions: string[] = [];

        for (let i = 0; i < options.length; i++) {
            const option = options[i];

            if (option.startsWith('--c=')) {
                leanOptions.push(option);
            } else if (option === inputFilename) {
                leanOptions.push(option);
            } else if (this.leanFlags.has(option)) {
                leanOptions.push(option);
            } else {
                leancOptions.push(option);
            }
        }

        return {leanOptions, leancOptions};
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const tmpDir = path.dirname(inputFilename);
        if (!execOptions.customCwd) {
            execOptions.customCwd = tmpDir;
        }

        const outputFilename = this.getOutputFilename(tmpDir, this.outputFilebase);

        const cOutputOption = options.find(option => option.startsWith('--c='));
        const cOutputFilename = cOutputOption?.substring('--c='.length) || this.getLeanCOutputFilename(outputFilename);

        const {leanOptions, leancOptions} = this.partitionOptionsForSteps(options, inputFilename);

        const leanResult = await this.doLeanBuildstep(compiler, leanOptions, execOptions, inputFilename);

        if (leanResult.code !== 0) {
            return {
                ...leanResult,
                inputFilename,
                languageId: 'asm',
            };
        }

        const leanc = await this.getLeancPath(compiler, execOptions);
        const leancResult = await this.doBuildstep(
            leanc,
            [...leancOptions, '-S', cOutputFilename, '-o', outputFilename],
            execOptions,
        );

        return {
            ...leancResult,
            okToCache: leanResult.okToCache && leancResult.okToCache,
            stdout: leanResult.stdout.concat(leancResult.stdout),
            stderr: leanResult.stderr.concat(leancResult.stderr),
            inputFilename,
            languageId: 'asm',
        };
    }
}
