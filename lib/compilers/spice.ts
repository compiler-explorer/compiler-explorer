// Copyright (c) 2024, Marc Auberer
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

import Semver from 'semver';
import _ from 'underscore';

import {ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {asSafeVer} from '../utils.js';

export class SpiceCompiler extends BaseCompiler {
    optLevelSuffix = '';

    static get key() {
        return 'spice';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIntel = true;
        this.compiler.supportsIrView = true;

        // The --abort-after-dump option was introduced in v0.19.3. Do not use it before that.
        if (Semver.lte(asSafeVer(this.compiler.semver), '0.19.2', true)) {
            this.compiler.irArg = ['-ir'];
            this.compiler.minIrArgs = ['-ir'];
        } else {
            this.compiler.irArg = ['-ir', '--abort-after-dump'];
            this.compiler.minIrArgs = ['-ir', '--abort-after-dump'];
        }

        this.compiler.optPipeline = {
            arg: ['-llvm', '-print-after-all', '-llvm', '-print-before-all'],
            moduleScopeArg: ['-llvm', '-print-module-scope'],
            noDiscardValueNamesArg: [],
        };
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions: string[],
    ): string[] {
        const options = ['build', '-g', '-o', outputFilename, '--dump-to-files', '-asm', '--abort-after-dump'];

        if (filters.intel) {
            options.push('-llvm', '--x86-asm-syntax=intel');
        }

        if (filters.binary || filters.binaryObject) {
            options.push('-obj');
        }

        this.optLevelSuffix = '';
        for (const item of userOptions) {
            if (item.startsWith('-O')) {
                if (item === '-O0') {
                    this.optLevelSuffix = '';
                } else if (item === '-Os') {
                    this.optLevelSuffix = '-O4';
                } else if (item === '-Oz') {
                    this.optLevelSuffix = '-O5';
                } else {
                    this.optLevelSuffix = item;
                }
            }
        }

        return options;
    }

    override getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const opts = super.getDefaultExecOptions();
        opts.env.SPICE_STD_DIR = path.join(path.dirname(this.compiler.exe), 'std');
        return opts;
    }

    override runExecutable(executable: string, executeParameters: ExecutableExecutionOptions, homeDir: string) {
        return super.runExecutable(executable, executeParameters, homeDir);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, 'example-assembly-code.s');
    }

    override getIrOutputFilename(inputFilename: string): string {
        const dirPath = path.dirname(inputFilename);
        return path.join(dirPath, 'example-ir-code' + this.optLevelSuffix + '.ll');
    }

    override getObjdumpOutputFilename(inputFilename: string): string {
        const dirPath = path.dirname(inputFilename);
        return path.join(dirPath, this.outputFilebase);
    }

    override filterUserOptions(userOptions: string[]): string[] {
        const forbiddenOptions = /^(((--(output|target))|(-o)|install|uninstall|test).*)$/;
        return _.filter(userOptions, (option: string) => !forbiddenOptions.test(option));
    }

    override isCfgCompiler(): boolean {
        return true;
    }
}
