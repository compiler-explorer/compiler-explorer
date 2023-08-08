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

import _ from 'underscore';

import type {BuildResult} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {AsmRaw} from '../parsers/asm-raw.js';
import {fileExists} from '../utils.js';

import {BaseParser} from './argument-parsers.js';

export class AssemblyCompiler extends BaseCompiler {
    static get key() {
        return 'assembly';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.asm = new AsmRaw();
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getArgumentParser() {
        return BaseParser;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        filters.binary = true;
        return [];
    }

    getGeneratedOutputFilename(fn: string) {
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

    override getOutputFilename(dirPath: string) {
        return this.getGeneratedOutputFilename(path.join(dirPath, 'example.asm'));
    }

    async runReadelf(fullResult, objectFilename) {
        const execOptions = this.getDefaultExecOptions();
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
        const execOptions = this.getDefaultExecOptions();
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

    override getExecutableFilename(dirPath: string) {
        return path.join(dirPath, 'ce-asm-executable');
    }

    override async buildExecutableInFolder(key, dirPath: string): Promise<BuildResult> {
        const buildEnvironment = this.setupBuildEnvironment(key, dirPath, true);

        const writeSummary = await this.writeAllFiles(dirPath, key.source, key.files, key.filters);
        const inputFilename = writeSummary.inputFilename;

        const outputFilename = this.getExecutableFilename(dirPath);

        const buildFilters: ParseFiltersAndOutputOptions = Object.assign({}, key.filters);
        buildFilters.binary = true;
        buildFilters.execute = false;

        const overrides = this.sanitizeCompilerOverrides(key.backendOptions.overrides || []);

        const compilerArguments = _.compact(
            this.prepareArguments(
                key.options,
                buildFilters,
                key.backendOptions,
                inputFilename,
                outputFilename,
                key.libraries,
                overrides,
            ),
        );

        const execOptions = this.getDefaultExecOptions();
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries);

        const downloads = await buildEnvironment;
        const result = await this.buildExecutable(key.compiler.exe, compilerArguments, inputFilename, execOptions);

        const fullResult: BuildResult = {
            ...result,
            buildsteps: [],
            downloads,
            executableFilename: outputFilename,
            compilationOptions: compilerArguments,
        };

        const objectFilename = this.getOutputFilename(dirPath);
        if (objectFilename !== inputFilename && (await fileExists(objectFilename))) {
            const inputArch = await this.getArchitecture(fullResult, objectFilename);
            const ldResult = await this.runLinker(fullResult, inputArch, objectFilename, outputFilename);

            fullResult.stderr = fullResult.stderr.concat(ldResult.stderr);
        }

        return fullResult;
    }

    override checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters: ParseFiltersAndOutputOptions) {
        return this.postProcess(asmResult, outputFilename, filters);
    }

    override getObjdumpOutputFilename(defaultOutputFilename: string): string {
        return this.getGeneratedOutputFilename(defaultOutputFilename);
    }

    override isCfgCompiler(/* compilerVersion */) {
        return true;
    }
}
