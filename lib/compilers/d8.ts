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

import path from 'path';

import fs from 'fs-extra';
import _ from 'underscore';

import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler, SimpleOutputFilenameCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import '../global.js';

import {JavaCompiler} from './java.js';
import {KotlinCompiler} from './kotlin.js';

export class D8Compiler extends BaseCompiler implements SimpleOutputFilenameCompiler {
    static get key() {
        return 'android-d8';
    }

    lineNumberRegex: RegExp;
    methodEndRegex: RegExp;

    minApiArgRegex: RegExp;

    jvmSyspropArgRegex: RegExp;
    syspropArgRegex: RegExp;

    javaId: string;
    kotlinId: string;

    libPaths: string[];

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super({...compilerInfo}, env);

        this.lineNumberRegex = /^\s+\.line\s+(\d+).*$/;
        this.methodEndRegex = /^\s*\.end\smethod.*$/;

        this.minApiArgRegex = /^--min-api$/;

        this.jvmSyspropArgRegex = /^-J.*$/;
        this.syspropArgRegex = /^-D.*$/;

        this.javaId = this.compilerProps<string>(`group.${this.compiler.group}.javaId`);
        this.kotlinId = this.compilerProps<string>(`group.${this.compiler.group}.kotlinId`);

        this.libPaths = [];
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.dex`);
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

        // D8 should not run if initial compile stage failed, the JavaCompiler
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

        const files = await fs.readdir(preliminaryCompilePath, {encoding: 'utf8', recursive: true});
        const classFiles = files.filter(f => f.endsWith('.class'));
        const d8Options = [
            ...syspropOptions,
            '-cp',
            this.compiler.exe, // R8 jar.
            'com.android.tools.r8.D8', // Main class name for the D8 compiler.
            ...userOptions,
            ...this.getMinApiArgument(useDefaultMinApi),
            ...classFiles,
        ];
        const result = await this.exec(javaCompiler.javaRuntime, d8Options, execOptions);
        return {
            ...this.transformToCompilationResult(result, outputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
        };
    }

    async generateSmali(outputFilename: string, maxSize: number) {
        const dirPath = path.dirname(outputFilename);

        const javaCompiler = unwrap(
            global.handler_config.compileHandler.findCompiler('java', this.javaId),
        ) as JavaCompiler;

        // There is only one dex file for all classes.
        let files = await fs.readdir(dirPath);
        const dexFile = files.find(f => f.endsWith('.dex'));
        const baksmaliOptions = ['-jar', this.compiler.objdumper, 'd', `${dexFile}`, '--code-offsets', '-o', dirPath];
        await this.exec(javaCompiler.javaRuntime, baksmaliOptions, {
            maxOutput: maxSize,
            customCwd: dirPath,
        });

        // There is one smali file for each class.
        files = await fs.readdir(dirPath);
        const smaliFiles = files.filter(f => f.endsWith('.smali'));
        let objResult = '';
        for (const smaliFile of smaliFiles) {
            objResult = objResult.concat(fs.readFileSync(path.join(dirPath, smaliFile), 'utf8') + '\n\n');
        }
        return objResult;
    }

    override async objdump(outputFilename: string, result: any, maxSize: number) {
        const objResult = await this.generateSmali(outputFilename, maxSize);
        const asmResult: ParsedAsmResult = {
            asm: [
                {
                    text: objResult,
                },
            ],
        };

        result.asm = asmResult.asm;
        return result;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        filters.binary = true;
        return [];
    }

    // Map line numbers to lines.
    override async processAsm(result): Promise<ParsedAsmResult> {
        if (result.code !== 0) {
            return {asm: [{text: result.asm, source: null}]};
        }
        const segments: ParsedAsmResultLine[] = [];
        const asm = result.asm[0].text;

        let lineNumber;
        for (const l of asm.split(/\n/)) {
            if (this.lineNumberRegex.test(l)) {
                lineNumber = Number.parseInt(l.match(this.lineNumberRegex)[1]);
                segments.push({text: l, source: null});
            } else if (this.methodEndRegex.test(l)) {
                lineNumber = null;
                segments.push({text: l, source: null});
            } else {
                if (/\S/.test(l) && lineNumber) {
                    segments.push({text: l, source: {file: null, line: lineNumber}});
                } else {
                    segments.push({text: l, source: null});
                }
            }
        }
        return {asm: segments};
    }

    getClasspathArgument(): string[] {
        const libString = this.libPaths.join(':');
        return libString ? ['-cp', libString] : [''];
    }

    getMinApiArgument(useDefaultMinApi: boolean): string[] {
        return useDefaultMinApi ? ['--min-api', '27'] : [''];
    }

    override getIncludeArguments(libraries: SelectedLibraryVersion[], dirPath: string): string[] {
        this.libPaths = libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return [];
            return foundVersion.path;
        });
        return this.libPaths;
    }
}
