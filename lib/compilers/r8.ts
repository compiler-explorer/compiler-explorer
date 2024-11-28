// Copyright (c) 2024, Compiler Explorer Authors
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

import fs from 'fs-extra';
import _ from 'underscore';

import {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {SimpleOutputFilenameCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';

import '../global.js';
import {D8Compiler} from './d8.js';
import {JavaCompiler} from './java.js';
import {KotlinCompiler} from './kotlin.js';

export class R8Compiler extends D8Compiler implements SimpleOutputFilenameCompiler {
    static override get key() {
        return 'android-r8';
    }

    kotlinLibPath: string;

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super({...compilerInfo}, env);
        this.kotlinLibPath = this.compilerProps<string>(`group.${this.compiler.group}.kotlinLibPath`);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const preliminaryCompilePath = path.dirname(inputFilename);
        let outputFilename = '';
        let initialResult: CompilationResult | null = null;

        const javaCompiler = unwrap(
            global.handler_config.compileHandler.findCompiler('java', this.javaId),
        ) as JavaCompiler;

        // Instantiate Java or Kotlin compiler based on the current language.
        if (this.lang.id === 'android-java') {
            outputFilename = javaCompiler.getOutputFilename(preliminaryCompilePath);
            const javaOptions = _.compact(
                javaCompiler.prepareArguments(
                    this.getClasspathArgument(),
                    javaCompiler.getDefaultFilters(),
                    {}, // backendOptions
                    inputFilename,
                    outputFilename,
                    [], // libraries
                    [], // overrides
                ),
            );
            initialResult = await javaCompiler.runCompiler(
                javaCompiler.getInfo().exe,
                javaOptions,
                this.filename(inputFilename),
                javaCompiler.getDefaultExecOptions(),
            );
        } else if (this.lang.id === 'android-kotlin') {
            const kotlinCompiler = unwrap(
                global.handler_config.compileHandler.findCompiler('kotlin', this.kotlinId),
            ) as KotlinCompiler;
            outputFilename = kotlinCompiler.getOutputFilename(preliminaryCompilePath);
            const kotlinOptions = _.compact(
                kotlinCompiler.prepareArguments(
                    this.getClasspathArgument(),
                    kotlinCompiler.getDefaultFilters(),
                    {}, // backendOptions
                    inputFilename,
                    outputFilename,
                    [], // libraries
                    [], // overrides
                ),
            );
            initialResult = await kotlinCompiler.runCompiler(
                kotlinCompiler.getInfo().exe,
                kotlinOptions,
                this.filename(inputFilename),
                kotlinCompiler.getDefaultExecOptions(),
            );
        } else {
            logger.error('Language is neither android-java nor android-kotlin.');
        }

        // R8 should not run if initial compile stage failed, the JavaCompiler
        // result can be returned instead.
        if (initialResult && initialResult.code !== 0) {
            return initialResult;
        }

        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        let useDefaultMinApi = true;

        // The items in 'options' before the source file are user inputs.
        const sourceFileOptionIndex = options.findIndex(option => {
            return option.endsWith('.java') || option.endsWith('.kt');
        });
        let userOptions = options.slice(0, sourceFileOptionIndex);
        const syspropOptions: string[] = [];
        for (const option of userOptions) {
            if (this.minApiArgRegex.test(option)) {
                useDefaultMinApi = false;
            } else if (this.jvmSyspropArgRegex.test(option)) {
                syspropOptions.push(option.replace('-J', '-'));
            } else if (this.syspropArgRegex.test(option)) {
                syspropOptions.push(option);
            }
        }
        userOptions = userOptions.filter(
            option => !this.jvmSyspropArgRegex.test(option) && !this.syspropArgRegex.test(option),
        );

        const files = await fs.readdir(preliminaryCompilePath);
        const classFiles = files.filter(f => f.endsWith('.class'));
        const r8Options = [
            '-Dcom.android.tools.r8.enableKeepAnnotations=1',
            ...syspropOptions,
            '-cp',
            this.compiler.exe, // R8 jar.
            'com.android.tools.r8.R8',
            ...this.getProguardConfigArguments(execOptions.customCwd),
            ...this.getR8LibArguments(),
            ...userOptions,
            ...this.getMinApiArgument(useDefaultMinApi),
            ...classFiles,
        ];
        const result = await this.exec(javaCompiler.javaRuntime, r8Options, execOptions);
        return {
            ...this.transformToCompilationResult(result, outputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
        };
    }

    getR8LibArguments(): string[] {
        const libArgs: string[] = [];
        for (const libPath of this.libPaths) {
            libArgs.push('--lib', libPath);
        }
        if (this.lang.id === 'android-kotlin') {
            libArgs.push(
                '--lib',
                this.kotlinLibPath + '/kotlin-stdlib.jar',
                '--lib',
                this.kotlinLibPath + '/annotations-13.0.jar',
            );
        }
        return libArgs;
    }

    getProguardConfigArguments(dir: string): string[] {
        const proguardCfgArgs: string[] = [];
        const proguardCfgPath = `${dir}/proguard.cfg`;
        if (fs.existsSync(proguardCfgPath)) {
            proguardCfgArgs.push('--pg-conf', proguardCfgPath);
        }
        return proguardCfgArgs;
    }
}
