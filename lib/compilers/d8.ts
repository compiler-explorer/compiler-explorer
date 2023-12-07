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

import {logger} from '../logger.js';

import {BaseCompiler} from '../base-compiler.js';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

export class D8Compiler extends BaseCompiler {
    static get key() {
        return 'android-d8';
    }

    lineNumberRegex: RegExp;
    methodEndRegex: RegExp;

    javaId: string;
    kotlinId: string;

    javaPath: string;
    javacPath: string;
    kotlincPath: string;

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super({...compilerInfo}, env);

        this.lineNumberRegex = /^\s+\.line\s+(\d+).*$/;
        this.methodEndRegex = /^\s*\.end\smethod.*$/;

        this.javaId = this.compilerProps<string>(`compiler.${this.compiler.id}.javaId`);
        this.kotlinId = this.compilerProps<string>(`compiler.${this.compiler.id}.kotlinId`);

        this.javaPath = this.compilerProps<string>(`compiler.${this.compiler.id}.javaPath`);
        this.javacPath = this.compilerProps<string>(`compiler.${this.compiler.id}.javacPath`);
        this.kotlincPath = this.compilerProps<string>(`compiler.${this.compiler.id}.kotlincPath`);
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.dex`);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
    ): Promise<CompilationResult> {
        const preliminaryCompilePath = path.dirname(inputFilename);
        let outputFilename = '';

        // Instantiate Java or Kotlin compiler based on the current language.
        if (this.lang.id === 'android-java') {
            const javaCompiler = global.handlerConfig.compileHandler.findCompiler('java', this.javaId);
            outputFilename = javaCompiler.getOutputFilename(preliminaryCompilePath);
            const javaOptions = _.compact(
                javaCompiler.prepareArguments(
                    [''], // options
                    javaCompiler.getDefaultFilters(),
                    {}, // backendOptions
                    inputFilename,
                    outputFilename,
                    [], // libraries
                    [], // overrides
                ),
            );
            await javaCompiler.runCompiler(
                this.javacPath,
                javaOptions,
                this.filename(inputFilename),
                javaCompiler.getDefaultExecOptions(),
            );
        } else if (this.lang.id === 'android-kotlin') {
            const kotlinCompiler = global.handlerConfig.compileHandler.findCompiler('kotlin', this.kotlinId);
            outputFilename = kotlinCompiler.getOutputFilename(preliminaryCompilePath);
            const kotlinOptions = _.compact(
                kotlinCompiler.prepareArguments(
                    [''], // options
                    kotlinCompiler.getDefaultFilters(),
                    {}, // backendOptions
                    inputFilename,
                    outputFilename,
                    [], // libraries
                    [], // overrides
                ),
            );
            await kotlinCompiler.runCompiler(
                this.kotlincPath,
                kotlinOptions,
                this.filename(inputFilename),
                kotlinCompiler.getDefaultExecOptions(),
            );
        } else {
            logger.error('Language is neither android-java nor android-kotlin.');
        }

        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        // The items in 'options' before the source file are user inputs.
        const sourceFileOptionIndex = options.findIndex(option => {
            return option.endsWith('.java') || option.endsWith('.kt');
        });

        const files = await fs.readdir(preliminaryCompilePath);
        const classFiles = files.filter(f => f.endsWith('.class'));
        const d8Options = [
            '-cp',
            this.compiler.exe, // R8 jar.
            'com.android.tools.r8.D8', // Main class name for the D8 compiler.
            ...options.slice(0, sourceFileOptionIndex),
            ...classFiles,
        ];
        const result = await this.exec(this.javaPath, d8Options, execOptions);
        return {
            ...this.transformToCompilationResult(result, outputFilename),
            languageId: this.getCompilerResultLanguageId(),
        };
    }

    override async objdump(outputFilename, result: any, maxSize: number) {
        const dirPath = path.dirname(outputFilename);

        // There is only one dex file for all classes.
        let files = await fs.readdir(dirPath);
        const dexFile = files.find(f => f.endsWith('.dex'));
        const baksmaliOptions = ['-jar', this.compiler.objdumper, 'd', `${dexFile}`, '-o', dirPath];
        const baksmaliResult = await this.exec(this.javaPath, baksmaliOptions, {
            maxOutput: maxSize,
            customCwd: dirPath,
        });

        // There is one smali file for each class.
        files = await fs.readdir(dirPath);
        const smaliFiles = files.filter(f => f.endsWith('.smali'));
        let objResult = '';
        for (const smaliFile of smaliFiles) {
            objResult = objResult.concat(fs.readFileSync(path.join(dirPath, smaliFile), 'utf-8') + '\n\n');
        }

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
    override async processAsm(result) {
        const asm = result.asm[0].text;
        const segments: ParsedAsmResultLine[] = [];

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
}
