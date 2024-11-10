// Copyright (c) 2017, Compiler Explorer Authors
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

import type {ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import * as utils from '../utils.js';

import {PascalParser} from './argument-parsers.js';
import {PascalUtils} from './pascal-utils.js';

export class FPCCompiler extends BaseCompiler {
    static get key() {
        return 'pascal';
    }

    dprFilename: string;
    supportsOptOutput: boolean;
    nasmPath: string;
    pasUtils: PascalUtils;
    demangler: any | null = null;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        this.compileFilename = 'output.pas';
        this.dprFilename = 'prog.dpr';
        this.supportsOptOutput = false;
        this.nasmPath = this.compilerProps<string>('nasmpath');
        this.pasUtils = new PascalUtils();
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions) {
        // TODO: Pascal doesn't have a demangler exe, it's the only compiler that's weird like this
        this.demangler = new (unwrap(this.demanglerClass))(null as any, this);
        return this.asm.process(result.asm, filters);
    }

    override async postProcess(result, outputFilename: string, filters: ParseFiltersAndOutputOptions) {
        const userSourceFilename = result.inputFilename;
        const pasFilepath = path.join(result.dirPath, userSourceFilename);
        const asmDumpFilepath = pasFilepath.substring(0, pasFilepath.length - 3) + 's';
        return super.postProcess(result, asmDumpFilepath, filters);
    }

    isTheSameFileProbably(originalName: string, compareTo: string): boolean {
        return originalName === compareTo || `/${originalName}` === compareTo;
    }

    override postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        if (!result.okToCache) return result;

        if (unwrap(filters).binary) {
            for (let j = 0; j < result.asm.length; ++j) {
                this.demangler.addDemangleToCache(result.asm[j].text);
            }
        }

        for (let j = 0; j < result.asm.length; ++j) {
            result.asm[j].text = this.demangler.demangleIfNeeded(result.asm[j].text);
            if (
                result.asm[j].source &&
                result.asm[j].source.file &&
                !result.asm[j].source.mainsource &&
                this.isTheSameFileProbably(result.inputFilename, result.asm[j].source.file)
            ) {
                result.asm[j].source.mainsource = true;
            }
        }

        return result;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        let options = ['-g', '-al'];

        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }

        filters.preProcessLines = this.preProcessLines.bind(this);

        if (filters.binary) {
            filters.dontMaskFilenames = true;
        }

        return options;
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        const inputFilename = this.getMainSourceFilename(key.source);
        const baseFilename = inputFilename.substring(0, inputFilename.length - 4);

        return path.join(dirPath, `${baseFilename}.s`);
    }

    override getExecutableFilename(dirPath: string) {
        return path.join(dirPath, 'prog');
    }

    override async objdump(
        outputFilename: string,
        result: any,
        maxSize: number,
        intelAsm: boolean,
        demangle: boolean,
        staticReloc: boolean,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const dirPath = path.dirname(outputFilename);

        let execBinary = this.getExecutableFilename(dirPath);

        const inputExt = path.extname(result.inputFilename);
        if (inputExt.toLowerCase() === '.dpr') {
            execBinary = path.join(dirPath, path.basename(result.inputFilename, inputExt));
        }

        if (await utils.fileExists(execBinary)) {
            return super.objdump(execBinary, result, maxSize, intelAsm, demangle, staticReloc, dynamicReloc, filters);
        }

        return super.objdump(outputFilename, result, maxSize, intelAsm, demangle, staticReloc, dynamicReloc, filters);
    }

    static preProcessBinaryAsm(input: string) {
        // todo: write some pascal logic into external asm-parser to only include user code,
        //  this is currently a giant mess

        const preamble = 'Disassembly of section .text:';
        const disasmStart = input.indexOf(preamble);

        let newSource = input.substring(0, disasmStart + preamble.length + 2);

        let foundSourceAt = input.indexOf('/app/', disasmStart);
        while (foundSourceAt !== -1) {
            const endOfProc = input.indexOf('...', foundSourceAt);

            const lookback = input.lastIndexOf('>:', foundSourceAt);
            if (lookback === -1) break;
            const furtherLookback = input.lastIndexOf('\n', lookback);
            if (furtherLookback === -1) break;

            if (endOfProc === -1) {
                newSource = newSource + input.substring(furtherLookback) + '\n';
                break;
            } else {
                newSource = newSource + input.substring(furtherLookback, endOfProc + 3) + '\n';
            }

            foundSourceAt = input.indexOf('/app/', endOfProc + 3);
        }

        return newSource.replaceAll('/app//', '/app/');
    }

    override postProcessObjdumpOutput(output: string) {
        return FPCCompiler.preProcessBinaryAsm(output);
    }

    async saveDummyProjectFile(filename: string, unitName: string, unitPath: string) {
        await fs.writeFile(
            filename,
            // prettier-ignore
            'program prog;\n' +
            'uses ' + unitName + ' in \'' + unitPath + '\';\n' +
            'begin\n' +
            'end.\n',
        );
    }

    getMainSourceFilename(source: string) {
        let inputFilename;
        if (this.pasUtils.isProgram(source)) {
            inputFilename = this.pasUtils.getProgName(source) + '.dpr';
        } else {
            const unitName = this.pasUtils.getUnitname(source);
            if (unitName) {
                inputFilename = unitName + '.pas';
            } else {
                inputFilename = this.compileFilename;
            }
        }

        return inputFilename;
    }

    override async writeAllFiles(dirPath: string, source: string, files: any[], filters: ParseFiltersAndOutputOptions) {
        const inputFilename = path.join(dirPath, this.getMainSourceFilename(source));

        if (source !== '' || !files) {
            await fs.writeFile(inputFilename, source);
        }

        if (files && files.length > 0) {
            await this.writeMultipleFiles(files, dirPath);
        }

        return {
            inputFilename,
        };
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const alreadyHasDPR = path.extname(inputFilename).toLowerCase() === '.dpr';
        const dirPath = path.dirname(inputFilename);

        let projectFile = path.join(dirPath, this.dprFilename);
        if (alreadyHasDPR) {
            projectFile = inputFilename;
        }

        execOptions.customCwd = dirPath;
        if (this.nasmPath) {
            execOptions.env = _.clone(process.env) as Record<string, string>;
            execOptions.env.PATH = execOptions.env.PATH + ':' + this.nasmPath;
        }

        if (!alreadyHasDPR) {
            const unitFilepath = path.basename(inputFilename);
            const unitName = unitFilepath.replace(/.pas$/i, '');
            await this.saveDummyProjectFile(projectFile, unitName, unitFilepath);
        }

        options.pop();
        options.push('-FE' + dirPath, '-B', projectFile);

        return this.parseOutput(await this.exec(compiler, options, execOptions), inputFilename, dirPath);
    }

    parseOutput(result, inputFilename: string, tempPath: string) {
        const fileWithoutPath = path.basename(inputFilename);
        result.inputFilename = fileWithoutPath;
        result.stdout = utils.parseOutput(result.stdout, fileWithoutPath, tempPath);
        result.stderr = utils.parseOutput(result.stderr, fileWithoutPath, tempPath);
        return result;
    }

    override getArgumentParserClass() {
        return PascalParser;
    }

    getExtraAsmHint(asm: string, currentFileId: number) {
        if (asm.startsWith('# [')) {
            const bracketEndPos = asm.indexOf(']', 3);
            let valueInBrackets = asm.substring(3, bracketEndPos);
            const colonPos = valueInBrackets.indexOf(':');
            if (colonPos !== -1) {
                valueInBrackets = valueInBrackets.substring(0, colonPos - 1);
            }

            if (valueInBrackets.startsWith('/')) {
                valueInBrackets = valueInBrackets.substring(1);
            }

            if (Number.isNaN(Number(valueInBrackets))) {
                return `  .file ${currentFileId} "${valueInBrackets}"`;
            } else {
                return `  .loc ${currentFileId} ${valueInBrackets} 0`;
            }
        } else if (asm.startsWith('.Le')) {
            return '  .cfi_endproc';
        } else {
            return false;
        }
    }

    tryGetFilenumber(asm: string, files: Record<string, number>) {
        if (asm.startsWith('# [')) {
            const bracketEndPos = asm.indexOf(']', 3);
            let valueInBrackets = asm.substring(3, bracketEndPos);
            const colonPos = valueInBrackets.indexOf(':');
            if (colonPos !== -1) {
                valueInBrackets = valueInBrackets.substring(0, colonPos - 1);
            }

            if (valueInBrackets.startsWith('/')) {
                valueInBrackets = valueInBrackets.substring(1);
            }

            if (Number.isNaN(Number(valueInBrackets))) {
                if (!files[valueInBrackets]) {
                    let maxFileId = _.max(files);
                    if (maxFileId === -Infinity) {
                        maxFileId = 0;
                    }

                    files[valueInBrackets] = maxFileId + 1;
                    return maxFileId + 1;
                }
            }
        }

        return false;
    }

    preProcessLines(asmLines: string[]) {
        let i = 0;
        const files: Record<string, number> = {};
        let currentFileId = 1;

        while (i < asmLines.length) {
            const newFileId = this.tryGetFilenumber(asmLines[i], files);
            if (newFileId) currentFileId = newFileId;

            const extraHint = this.getExtraAsmHint(asmLines[i], currentFileId);
            if (extraHint) {
                i++;
                asmLines.splice(i, 0, extraHint);
            } else {
                this.demangler.addDemangleToCache(asmLines[i]);
            }

            i++;
        }

        return asmLines;
    }
}
