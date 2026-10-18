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

import Semver from 'semver';

import {splitArguments} from '../../shared/common-utils.js';
import type {
    ActiveTool,
    BypassCache,
    ExecutionParams,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {BasicExecutionResult, ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import type {IExecutionEnvironment} from '../execution/execution-env.interfaces.js';

export class MicroPythonCompiler extends BaseCompiler {
    static get key() {
        return 'micropython';
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.mpy`);
    }

    override prepareArguments(
        userOptions: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        inputFilename: string,
        outputFilename: string,
        libraries: SelectedLibraryVersion[],
        overrides: ConfiguredOverrides,
    ) {
        if (Semver.eq(this.compiler.semver, '1.20.0')) {
            return [
                ...['-o', outputFilename],
                ...(filters.dontMaskFilenames ? [] : ['-s', this.compileFilename]),
                ...(this.compiler.options ? splitArguments(this.compiler.options) : []),
                ...userOptions,
                ...[inputFilename],
            ];
        } else {
            return [
                ...['-o', outputFilename],
                ...(filters.dontMaskFilenames ? [] : ['-s', this.compileFilename]),
                ...(this.compiler.options ? splitArguments(this.compiler.options) : []),
                ...userOptions,
                ...['--', inputFilename],
            ];
        }
    }

    override async compile(
        source: string,
        options: string[],
        backendOptions: Record<string, any>,
        filters: ParseFiltersAndOutputOptions,
        bypassCache: BypassCache,
        tools: ActiveTool[],
        executeParameters: ExecutionParams,
        libraries: SelectedLibraryVersion[],
        files: FiledataPair[],
    ) {
        filters.binary = true;
        return super.compile(
            source,
            options,
            backendOptions,
            filters,
            bypassCache,
            tools,
            executeParameters,
            libraries,
            files,
        );
    }

    override async runExecutable(
        executable: string,
        executeParameters: ExecutableExecutionOptions,
        homeDir: string,
    ): Promise<BasicExecutionResult> {
        const execOptionsCopy: ExecutableExecutionOptions = JSON.parse(
            JSON.stringify(executeParameters),
        ) as ExecutableExecutionOptions;

        execOptionsCopy.args = [
            ...this.compiler.executionWrapperArgs,
            ...['-m', path.basename(executable, '.mpy')],
            ...execOptionsCopy.args,
        ];
        executable = this.compiler.executionWrapper;

        const execEnv: IExecutionEnvironment = new this.executionEnvironmentClass(this.env);
        return execEnv.execBinary(executable, execOptionsCopy, homeDir);
    }
}
