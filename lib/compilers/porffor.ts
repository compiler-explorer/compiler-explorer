// Copyright (c) 2025, Compiler Explorer Authors
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
import _ from 'underscore';

import {CacheKey, CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {PorfforParser} from './argument-parsers.js';

export class PorfforCompiler extends BaseCompiler {
    target = 'wasm';

    static get key(): string {
        return 'porffor';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.compiler.supportsIntel = true;
    }

    override async handleInterpreting(
        key: CacheKey,
        executeParameters: ExecutableExecutionOptions,
    ): Promise<CompilationResult> {
        // Prevent the interpreter (which uses node) from inheriting our NODE_OPTIONS.
        executeParameters.env.NODE_OPTIONS = '';

        return super.handleInterpreting(key, executeParameters);
    }

    override getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const opts = super.getDefaultExecOptions();
        opts.env.NODE_OPTIONS = '';

        return opts;
    }

    getTargetFromOptions(options: string[]): string {
        return options.find(o => o.startsWith('--target='))?.substring(9) ?? 'wasm';
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        const options = unwrap(userOptions);

        this.target = this.getTargetFromOptions(options);
        if (this.target === 'wasm' || this.target == 'native') filters.binary = true;

        return ['-d', `-o=${this.filename(outputFilename)}`];
    }

    getOutputExtension(target: string): string {
        switch (target) {
            case 'c': {
                return '.c';
            }
            case 'native': {
                return '';
            }
            default: {
                return '.wasm';
            }
        }
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey): string {
        return path.join(dirPath, `out${this.getOutputExtension(this.target)}`);
    }

    override isCfgCompiler(): boolean {
        return true;
    }

    override getArgumentParserClass() {
        return PorfforParser;
    }

    override getSharedLibraryPathsAsArguments(libraries: SelectedLibraryVersion[], libDownloadPath?: string): string[] {
        return [];
    }

    override getSharedLibraryLinks(libraries: SelectedLibraryVersion[]): string[] {
        return [];
    }
}
