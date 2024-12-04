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

import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {logger} from '../logger.js';
import {SPIRVAsmParser} from '../parsers/asm-parser-spirv.js';
import * as utils from '../utils.js';

export class SlangCompiler extends BaseCompiler {
    protected disassemblerPath: string;
    protected spirvAsm: SPIRVAsmParser;

    static get key() {
        return 'slang';
    }

    constructor(info: any, env: any) {
        super(info, env);

        this.spirvAsm = new SPIRVAsmParser(this.compilerProps);

        this.disassemblerPath = this.compilerProps<string>('disassemblerPath');
    }

    getTarget(options?: string[]) {
        if (options) {
            const index = options.indexOf('-target');
            if (index !== -1) {
                return options[index + 1];
            }
        }
        return 'spirv'; // no target found, slang default to 'spirv'
    }

    getPrimaryOutputFilename(dirPath: string, outputFilebase: string, target: string) {
        if (target === 'spirv') {
            return path.join(dirPath, `${outputFilebase}.spv`);
        } else {
            // If there is no intermediate file needed, can use file output
            return this.getOutputFilename(dirPath, outputFilebase);
        }
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        const sourceDir = path.dirname(outputFilename);
        const target = this.getTarget(userOptions);
        const spvBinFilename = this.getPrimaryOutputFilename(sourceDir, this.outputFilebase, target);
        return ['-o', spvBinFilename, '-gminimal']; // -g provides debug info
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, `${outputFilebase}.spvasm`);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
    ) {
        const sourceDir = path.dirname(inputFilename);
        const target = this.getTarget(options);
        const slangOutputFilename = this.getPrimaryOutputFilename(sourceDir, this.outputFilebase, target);

        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        execOptions.customCwd = path.dirname(inputFilename);

        const slangOutput = await this.exec(compiler, options, execOptions);
        const result = this.transformToCompilationResult(slangOutput, inputFilename);

        if (slangOutput.code !== 0 || !(await utils.fileExists(slangOutputFilename))) {
            return result;
        }

        // While slang does have a way to dump spirv assembly, using spirv-dis will make the SPIR-V we
        // display unified with other things that produce SPIR-V
        if (target === 'spirv') {
            const spvasmFilename = this.getOutputFilename(sourceDir, this.outputFilebase);
            const disassemblerFlags = [slangOutputFilename, '-o', spvasmFilename, '--comment'];

            const spvasmOutput = await this.exec(this.disassemblerPath, disassemblerFlags, execOptions);
            if (spvasmOutput.code !== 0) {
                logger.error('SPIR-V binary to text failed', spvasmOutput);
            }

            result.stdout = result.stdout.concat(utils.parseOutput(spvasmOutput.stdout));
            result.stderr = result.stderr.concat(utils.parseOutput(spvasmOutput.stderr));
            result.languageId = 'spirv';
        }

        return result;
    }

    // slangc defaults to SPIR-V as a target, but has many target it can output
    override async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        if (result.asm.startsWith('; SPIR-V')) {
            return this.spirvAsm.processAsm(result.asm, filters);
        }
        // If not SPIR-V, just display as plain text to be safe
        return super.processAsm(result, filters, options);
    }
}
