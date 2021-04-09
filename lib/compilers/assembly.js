// Copyright (c) 2018, Compiler Explorer Authors
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

import fs from 'fs';
import path from 'path';

import { AsmRaw } from '../asm-raw';
import { BaseCompiler } from '../base-compiler';
import * as utils from '../utils';

import { BaseParser } from './argument-parsers';

export class AssemblyCompiler extends BaseCompiler {
    static get key() { return 'assembly'; }

    constructor(info, env) {
        super(info, env);
        this.asm = new AsmRaw();
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    getArgumentParser() {
        return BaseParser;
    }

    optionsForFilter(filters) {
        filters.binary = true;
        return [];
    }

    getGeneratedOutputFilename(fn) {
        const outputFolder = path.dirname(fn);
        const files = fs.readdirSync(outputFolder);

        let outputFilename = super.filename(fn);
        files.forEach(file => {
            if (file[0] !== '.' && file !== this.compileFilename) {
                outputFilename = path.join(outputFolder, file);
            }
        });

        return outputFilename;
    }

    getOutputFilename(dirPath) {
        return this.getGeneratedOutputFilename(path.join(dirPath, 'example.asm'));
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        execOptions.customCwd = path.dirname(inputFilename);

        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;
        result.stdout = utils.parseOutput(result.stdout, inputFilename);
        result.stderr = utils.parseOutput(result.stderr, inputFilename);
        return result;
    }

    async getArchitecture(objectFilename) {
        let execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = path.dirname(objectFilename);

        const result = await this.exec(this.env.ceProps('readelf'), ['-h', objectFilename], execOptions);
        if (result.stdout.includes('ELF32') && result.stdout.includes('80386')) {
            return 'x86';
        } else if (result.stdout.includes('ELF64') && result.stdout.includes('X86-64')) {
            return 'x86_64';
        }

        return false;
    }

    async runLinker(inputFilename, outputFilename) {
        let execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = path.dirname(inputFilename);

        const options = ['-o', outputFilename];
        const inputArch = await this.getArchitecture(inputFilename);
        if (inputArch === 'x86') {
            options.push('-m');
            options.push('elf_i386');
        } else if (inputArch === 'x86_64') {
            // default target
        } else {
            return {
                stderr: utils.parseOutput('Unknown architecture'),
            };
        }
        options.push(inputFilename);

        const result = await this.exec(this.env.ceProps('ld'), options, execOptions);
        result.inputFilename = inputFilename;
        result.stdout = utils.parseOutput(result.stdout, inputFilename);
        result.stderr = utils.parseOutput(result.stderr, inputFilename);
        return result;
    }

    getExecutableFilename(dirPath) {
        return path.join(dirPath, 'a.out');
    }

    async buildExecutable(compiler, options, inputFilename, execOptions) {
        const compilationResult = await this.runCompiler(compiler, options, inputFilename, execOptions);

        const dirPath = path.dirname(inputFilename);
        const executableFilename = this.getExecutableFilename(dirPath);
        const objectFile = this.getOutputFilename(dirPath);
        const linkResult = await this.runLinker(objectFile, executableFilename);

        return Object.assign({}, compilationResult, linkResult);
    }

    checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        return this.postProcess(asmResult, outputFilename, filters);
    }

    getObjdumpOutputFilename(defaultOutputFilename) {
        return this.getGeneratedOutputFilename(defaultOutputFilename);
    }
}
