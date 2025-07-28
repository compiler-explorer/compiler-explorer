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

import {createWriteStream} from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';

import type {
    CompilationResult,
    ExecutionOptions,
    ExecutionOptionsWithEnv,
} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import * as exec from '../exec.js';
import {logger} from '../logger.js';
import * as utils from '../utils.js';

// AMD RGA := AMD's Radeon GPU Analyzer (https://gpuopen.com/rga/)
export class AMDRGACompiler extends BaseCompiler {
    static get key() {
        return 'amd_rga';
    }

    constructor(info: any, env: any) {
        super(info, env);
        this.compiler.supportsIntel = false;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: any, userOptions?: any): any[] {
        return [outputFilename];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const result = await this.execRGA(compiler, options, execOptions);
        return this.transformToCompilationResult(result, inputFilename);
    }

    async execRGA(filepath: string, args: string[], execOptions: ExecutionOptions): Promise<any> {
        // Track the total time spent instead of relying on executeDirect's internal timing facility
        const startTime = process.hrtime.bigint();

        // If help flag is requested, skip all extra options
        if (args.includes('-h') || args.includes('--help')) {
            logger.debug(`RGA help mode. Args: ${args.join(' ')}`);
            return await exec.execute(filepath, args, execOptions);
        }

        const outputIsaFile = args[0];
        const outputDir = path.dirname(outputIsaFile);

        const rgaArgs = ['--isa', outputIsaFile, ...args.slice(1)];

        // Pop the last argument assuming it's the HLSL input file
        const inputHlslFile = rgaArgs.pop()!;

        // Determine which flag to use based on RGA shader mode.
        if (args.includes('-s') && args[args.indexOf('-s') + 1] === 'dxr') {
            rgaArgs.push('--hlsl');
        } else if (args.includes('-s') && args[args.indexOf('-s') + 1] === 'dx12') {
            rgaArgs.push('--all-hlsl');
        }

        rgaArgs.push(inputHlslFile, '--offline');
        logger.debug(`RGA args: ${rgaArgs}`);

        const rgaResult = await exec.execute(filepath, rgaArgs, execOptions);
        if (rgaResult.code !== 0) {
            // Failed to compile AMD ISA
            const endTime = process.hrtime.bigint();
            rgaResult.execTime = utils.deltaTimeNanoToMili(startTime, endTime);
            return rgaResult;
        }

        // RGA doesn't emit the exact file requested. Append all files with the same extension
        // as outputIsaFile into a new file outputIsaFile in outputDir.
        const files = await fs.readdir(outputDir, {encoding: 'utf8'});
        const outputIsaFileExt = path.extname(outputIsaFile);
        const outputIsaFileBase = path.basename(outputIsaFile, outputIsaFileExt);
        const writeStream = createWriteStream(outputIsaFile);

        for (const file of files) {
            if (file.endsWith(outputIsaFileExt) && file.includes(outputIsaFileBase)) {
                const filePath = path.join(outputDir, file);
                const content = await fs.readFile(filePath, 'utf8');
                writeStream.write(content);
                writeStream.write('\n\n');
            }
        }
        writeStream.end();

        const endTime = process.hrtime.bigint();
        rgaResult.execTime = utils.deltaTimeNanoToMili(startTime, endTime);
        return rgaResult;
    }
}
