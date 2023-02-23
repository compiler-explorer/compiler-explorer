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

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {unwrap} from '../assert';
import {BaseCompiler} from '../base-compiler';
import * as utils from '../utils';

import {PascalParser} from './argument-parsers';
import {PascalUtils} from './pascal-utils';

export class FPCCompiler extends BaseCompiler {
    static get key() {
        return 'pascal';
    }

    dprFilename: string;
    supportsOptOutput: boolean;
    nasmPath: string;
    pasUtils: PascalUtils;
    demangler: any | null = null;

    constructor(info: PreliminaryCompilerInfo, env) {
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

    override processAsm(result, filters) {
        // TODO: Pascal doesn't have a demangler exe, it's the only compiler that's weird like this
        this.demangler = new (unwrap(this.demanglerClass))(null as any, this);
        return this.asm.process(result.asm, filters);
    }

    override postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        if (!result.okToCache) return result;

        if (unwrap(filters).binary) {
            for (let j = 0; j < result.asm.length; ++j) {
                this.demangler.addDemangleToCache(result.asm[j].text);
            }
        }

        for (let j = 0; j < result.asm.length; ++j)
            result.asm[j].text = this.demangler.demangleIfNeeded(result.asm[j].text);

        return result;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        let options = ['-g', '-al'];

        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }

        filters.preProcessLines = _.bind(this.preProcessLines, this);

        if (filters.binary) {
            filters.dontMaskFilenames = true;
        }

        return options;
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.s`);
    }

    override getExecutableFilename(dirPath: string) {
        return path.join(dirPath, 'prog');
    }

    override async objdump(
        outputFilename,
        result: any,
        maxSize: number,
        intelAsm,
        demangle,
        staticReloc: boolean,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions
    ) {
        const dirPath = path.dirname(outputFilename);
        const execBinary = this.getExecutableFilename(dirPath);
        if (await utils.fileExists(execBinary)) {
            return super.objdump(execBinary, result, maxSize, intelAsm, demangle, staticReloc, dynamicReloc, filters);
        }

        return super.objdump(outputFilename, result, maxSize, intelAsm, demangle, staticReloc, dynamicReloc, filters);
    }

    static preProcessBinaryAsm(input: string) {
        const systemInitOffset = input.indexOf('<SYSTEM_$$_init$>');
        const relevantAsmStartsAt = input.indexOf('...', systemInitOffset);
        if (relevantAsmStartsAt !== -1) {
            const lastLinefeedBeforeStart = input.lastIndexOf('\n', relevantAsmStartsAt);
            if (lastLinefeedBeforeStart === -1) {
                input = input.substr(0, input.indexOf('00000000004')) + '\n' + input.substr(relevantAsmStartsAt);
            } else {
                input =
                    input.substr(0, input.indexOf('00000000004')) + '\n' + input.substr(lastLinefeedBeforeStart + 1);
            }
        }
        return input;
    }

    override postProcessObjdumpOutput(output) {
        return FPCCompiler.preProcessBinaryAsm(output);
    }

    async saveDummyProjectFile(filename: string, unitName: string, unitPath: string) {
        await fs.writeFile(
            filename,
            // prettier-ignore
            'program prog;\n' +
            'uses ' + unitName + ' in \'' + unitPath + '\';\n' +
            'begin\n' +
            'end.\n'
        );
    }

    override async writeAllFiles(dirPath: string, source: string, files: any[], filters: ParseFiltersAndOutputOptions) {
        let inputFilename;
        if (this.pasUtils.isProgram(source)) {
            inputFilename = path.join(dirPath, this.dprFilename);
        } else {
            const unitName = this.pasUtils.getUnitname(source);
            if (unitName) {
                inputFilename = path.join(dirPath, unitName + '.pas');
            } else {
                inputFilename = path.join(dirPath, this.compileFilename);
            }
        }

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
        execOptions: ExecutionOptions
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const alreadyHasDPR = path.basename(inputFilename) === this.dprFilename;
        const dirPath = path.dirname(inputFilename);

        const projectFile = path.join(dirPath, this.dprFilename);
        execOptions.customCwd = dirPath;
        if (this.nasmPath) {
            execOptions.env = _.clone(process.env);
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

    override getArgumentParser() {
        return PascalParser;
    }

    getExtraAsmHint(asm: string, currentFileId: number) {
        if (asm.startsWith('# [')) {
            const bracketEndPos = asm.indexOf(']', 3);
            let valueInBrackets = asm.substr(3, bracketEndPos - 3);
            const colonPos = valueInBrackets.indexOf(':');
            if (colonPos !== -1) {
                valueInBrackets = valueInBrackets.substr(0, colonPos - 1);
            }

            if (valueInBrackets.startsWith('/')) {
                valueInBrackets = valueInBrackets.substr(1);
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
            let valueInBrackets = asm.substr(3, bracketEndPos - 3);
            const colonPos = valueInBrackets.indexOf(':');
            if (colonPos !== -1) {
                valueInBrackets = valueInBrackets.substr(0, colonPos - 1);
            }

            if (valueInBrackets.startsWith('/')) {
                valueInBrackets = valueInBrackets.substr(1);
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
