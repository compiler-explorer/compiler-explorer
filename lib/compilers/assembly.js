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

import {BaseCompiler} from '../base-compiler';
import {AsmRaw} from '../parsers/asm-raw';
import * as utils from '../utils';

import {BaseParser} from './argument-parsers';

export class AssemblyCompiler extends BaseCompiler {
    static get key() {
        return 'assembly';
    }

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
        for (const file of files) {
            if (file[0] !== '.' && file !== this.compileFilename) {
                outputFilename = path.join(outputFolder, file);
            }
        }

        return outputFilename;
    }

    getOutputFilename(dirPath) {
        return this.getGeneratedOutputFilename(path.join(dirPath, 'example.asm'));
    }

    async runReadelf(fullResult, objectFilename) {
        let execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = path.dirname(objectFilename);
        return await this.doBuildstepAndAddToResult(
            fullResult,
            'readelf',
            this.env.ceProps('readelf'),
            ['-h', objectFilename],
            execOptions,
        );
    }

    async getArchitecture(fullResult, objectFilename) {
        const result = await this.runReadelf(fullResult, objectFilename);
        const output = result.stdout.map(line => line.text).join('\n');
        if (output.includes('ELF32') && output.includes('80386')) {
            return 'x86';
        } else if (output.includes('ELF64') && output.includes('X86-64')) {
            return 'x86_64';
        } else if (output.includes('Mach-O 64-bit x86-64')) {
            // note: this is to support readelf=objdump on Mac
            return 'x86_64';
        }

        return false;
    }

    async runLinker(fullResult, inputArch, objectFilename, outputFilename) {
        let execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = path.dirname(objectFilename);

        const options = ['-o', outputFilename];
        if (inputArch === 'x86') {
            options.push('-m', 'elf_i386');
        } else if (inputArch === 'x86_64') {
            // default target
        } else {
            const result = {
                code: -1,
                step: 'ld',
                stderr: [{text: 'Invalid architecture for linking and execution'}],
            };
            fullResult.buildsteps.push(result);
            return result;
        }
        options.push(objectFilename);

        return this.doBuildstepAndAddToResult(fullResult, 'ld', this.env.ceProps('ld'), options, execOptions);
    }

    getExecutableFilename(dirPath) {
        return path.join(dirPath, 'ce-asm-executable');
    }

    async buildExecutable(compiler, options, inputFilename, execOptions) {
        const fullResult = {
            code: -1,
            executableFilename: '',
            buildsteps: [],
        };

        const compilationResult = await this.runCompiler(compiler, options, inputFilename, execOptions);
        compilationResult.step = 'Assembling';
        fullResult.buildsteps.push(compilationResult);

        const dirPath = path.dirname(inputFilename);
        fullResult.executableFilename = this.getExecutableFilename(dirPath);
        const objectFilename = this.getOutputFilename(dirPath);

        const inputArch = await this.getArchitecture(fullResult, objectFilename);
        const ldResult = await this.runLinker(fullResult, inputArch, objectFilename, fullResult.executableFilename);

        fullResult.code = ldResult.code;
        if (ldResult.stderr.length > 0) {
            fullResult.stderr = ldResult.stderr;
        }

        return fullResult;
    }

    checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        return this.postProcess(asmResult, outputFilename, filters);
    }

    getObjdumpOutputFilename(defaultOutputFilename) {
        return this.getGeneratedOutputFilename(defaultOutputFilename);
    }
}
