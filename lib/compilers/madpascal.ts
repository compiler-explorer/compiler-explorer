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

import {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {ArtifactType} from '../../types/tool.interfaces.js';
import {addArtifactToResult} from '../artifact-utils.js';
import {BaseCompiler, c_value_placeholder} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {MadsAsmParser} from '../parsers/asm-parser-mads.js';
import * as utils from '../utils.js';

import {MadpascalParser} from './argument-parsers.js';

export class MadPascalCompiler extends BaseCompiler {
    protected madsExe: any;

    static get key() {
        return 'madpascal';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.compileFilename = 'output.pas';
        this.madsExe = this.compilerProps<string>(`compiler.${info.id}.madsexe`);
        this.asm = new MadsAsmParser(this.compilerProps);
    }

    getCompilerOutputFilename(dirPath: string, outputFilebase: string) {
        const filename = `${outputFilebase}.a65`;
        if (dirPath) {
            return path.join(dirPath, filename);
        } else {
            return filename;
        }
    }

    getAssemblerOutputFilename(dirPath: string, outputFilebase: string) {
        const filename = `${outputFilebase}.obx`;
        if (dirPath) {
            return path.join(dirPath, filename);
        } else {
            return filename;
        }
    }

    getListingFilename(dirPath: string, outputFilebase: string) {
        const filename = `${outputFilebase}.lst`;
        if (dirPath) {
            return path.join(dirPath, filename);
        } else {
            return filename;
        }
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return this.getCompilerOutputFilename(dirPath, outputFilebase);
    }

    protected override getArgumentParserClass(): any {
        return MadpascalParser;
    }

    protected override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        filters.demangle = false;
        return [];
    }

    isTargettingC64(options: string[]) {
        if (options.includes('-target:c64')) return true;

        const p = options.indexOf('-t');
        if (p !== -1) {
            return options[p + 1] === 'c64';
        }

        return false;
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

        const tmpDir = path.dirname(inputFilename);
        if (!execOptions.customCwd) {
            execOptions.customCwd = tmpDir;
        }

        const compileResult = await this.exec(compiler, options, execOptions);

        const result = {
            ...this.transformToCompilationResult(compileResult, inputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
        };

        const outputFilename = this.getCompilerOutputFilename(tmpDir, this.outputFilebase);

        if (filters?.binary && (await utils.fileExists(outputFilename))) {
            const compilerDir = path.dirname(compiler);
            const baseDir = path.join(compilerDir, '../base');
            const assemblerResult = await this.exec(
                this.madsExe,
                [outputFilename, '-p', '-x', `-i:${baseDir}`],
                execOptions,
            );

            if (assemblerResult.code === 0 && this.isTargettingC64(options)) {
                const diskfile = path.join(tmpDir, 'output.obx');
                if (await utils.fileExists(diskfile)) {
                    await addArtifactToResult(result, diskfile, ArtifactType.c64prg, 'output.prg');
                }
            }

            return {
                ...result,
                ...this.transformToCompilationResult(assemblerResult, inputFilename),
                languageId: this.getCompilerResultLanguageId(filters),
            };
        }

        return result;
    }

    override async objdump(
        outputFilename: string,
        result: any,
        maxSize: number,
        intelAsm: boolean,
        demangle: boolean,
        staticReloc: boolean | undefined,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const tmpDir = path.dirname(outputFilename);
        const listingFilename = this.getListingFilename(tmpDir, this.outputFilebase);

        if (!(await utils.fileExists(listingFilename))) {
            result.asm = '<No output file ' + listingFilename + '>';
            return result;
        }

        const content = await fs.readFile(listingFilename);
        result.asm = this.postProcessObjdumpOutput(content.toString('utf8'));

        return result;
    }

    override getTargetFlags(): string[] {
        return [`-target:${c_value_placeholder}`];
    }

    override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        return options.concat([this.filename(inputFilename)], libIncludes, libOptions, userOptions);
    }
}
