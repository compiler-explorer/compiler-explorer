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

import fs from 'node:fs/promises';
import path from 'node:path';

import _ from 'underscore';

import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import * as utils from '../utils.js';
import {ClojureParser} from './argument-parsers.js';
import {JavaCompiler} from './java.js';

export class ClojureCompiler extends JavaCompiler {
    compilerWrapperPath: string;
    defaultDeps: string;
    configDir: string;
    javaHome: string;

    static override get key() {
        return 'clojure';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        // Use invalid Clojure filename to avoid clashing with name determined by namespace
        this.compileFilename = `example-source${this.lang.extensions[0]}`;
        this.javaHome = this.compilerProps<string>(`compiler.${this.compiler.id}.java_home`);
        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper', '') ||
            utils.resolvePathFromAppRoot('etc', 'scripts', 'clojure_wrapper.clj');
        this.compiler.supportsClojureMacroExpView = true;
        this.configDir =
            this.compilerProps<string>(`compiler.${this.compiler.id}.config_dir`) ||
            path.resolve(path.dirname(this.compiler.exe), '../.config');
        const repoDir =
            this.compilerProps<string>(`compiler.${this.compiler.id}.repo_dir`) ||
            path.resolve(path.dirname(this.compiler.exe), '../.m2/repository');
        this.defaultDeps = `{:mvn/local-repo "${repoDir}"}`;
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.javaHome) {
            execOptions.env.JAVA_HOME = this.javaHome;
        }
        execOptions.env.CLJ_CONFIG = this.configDir;

        return execOptions;
    }

    override filterUserOptions(userOptions: string[]) {
        return userOptions.filter(option => {
            // Filter out anything that looks like a Clojure source file
            // that would confuse the wrapper.
            // Also, don't allow users to specify macro expansion mode used
            // internally.
            return !option.match(/^.*\.clj$/) && option !== '--macro-expand';
        });
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        // Forcibly enable javap
        filters.binary = true;
        return [];
    }

    override getArgumentParserClass() {
        return ClojureParser;
    }

    override async readdir(dirPath: string): Promise<string[]> {
        // Clojure requires recursive walk to find namespace-pathed class files
        return fs.readdir(dirPath, {recursive: true});
    }

    async getClojureClasspathArgument(
        dirPath: string,
        compiler: string,
        execOptions: ExecutionOptionsWithEnv,
    ): Promise<string[]> {
        const pathOption = ['-Sdeps', this.defaultDeps, '-Spath'];
        const output = await this.exec(compiler, pathOption, execOptions);
        const cp = dirPath + ':' + output.stdout.trim();
        return ['-Scp', cp];
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
        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        // The items in 'options' before the source file are user inputs.
        const sourceFileOptionIndex = options.findIndex(option => {
            return option.endsWith('.clj');
        });
        const userOptions = options.slice(0, sourceFileOptionIndex);
        const classpathArgument = await this.getClojureClasspathArgument(execOptions.customCwd, compiler, execOptions);
        const wrapperInvokeArgument = ['-M', this.compilerWrapperPath];
        const clojureOptions = _.compact([
            '-Sdeps',
            this.defaultDeps,
            ...classpathArgument,
            ...wrapperInvokeArgument,
            ...userOptions,
            inputFilename,
        ]);
        const result = await this.exec(compiler, clojureOptions, execOptions);
        return {
            ...this.transformToCompilationResult(result, inputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
            instructionSet: this.getInstructionSetFromCompilerArgs(options),
        };
    }

    override async generateClojureMacroExpansion(inputFilename: string, options: string[]): Promise<ResultLine[]> {
        // The items in 'options' before the source file are user inputs.
        const sourceFileOptionIndex = options.findIndex(option => {
            return option.endsWith('.clj');
        });
        const userOptions = options.slice(0, sourceFileOptionIndex);
        const clojureOptions = _.compact([...userOptions, '--macro-expand', inputFilename]);
        const output = await this.runCompiler(
            this.compiler.exe,
            clojureOptions,
            inputFilename,
            this.getDefaultExecOptions(),
        );
        if (output.code !== 0) {
            return [{text: `Failed to run compiler to get Clojure Macro Expansion`}, ...output.stderr];
        }
        return output.stdout;
    }
}
