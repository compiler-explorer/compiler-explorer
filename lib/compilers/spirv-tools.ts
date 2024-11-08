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

import type {ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {SPIRVAsmParser} from '../parsers/asm-parser-spirv.js';
import * as utils from '../utils.js';

// If you want to output SPIR-V, most likely you want SPIRVAsmParser
//
// SPIR-V is an IR that targets heterogeneous (like a GPU) and has a "compute" and "graphics" mode.
// When used for graphics, it inself is not enough info to "compile" down to the GPU's assembly because normally there
// is other graphics // pipeline information that is required (ex. is the depth test enable/disabled)
//
// SPIR-V has a lot of tooling around it to optimize, validate, fuzz, etc. This compiler is only used for tooling.
export class SPIRVToolsCompiler extends BaseCompiler {
    protected assemblerPath: string;
    protected disassemblerPath: string;
    protected spirvAsm: SPIRVAsmParser;

    static get key() {
        return 'spirv-tools';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.spirvAsm = new SPIRVAsmParser(this.compilerProps);

        // spirv-as
        this.assemblerPath = this.compilerProps<string>('assemblerPath');
        // spirv-dis
        this.disassemblerPath = this.compilerProps<string>('disassemblerPath');
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const sourceDir = path.dirname(outputFilename);
        const spvBinFilename = this.getPrimaryOutputFilename(sourceDir, this.outputFilebase);
        return ['-o', spvBinFilename];
    }

    getPrimaryOutputFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, `${outputFilebase}.spv`);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, `${outputFilebase}.spvasm`);
    }

    // Some tools (ex. spirv-opt) needs this as a single argument, so merge with "="
    mergeSpirvTargetEnv(options: string[]) {
        const index = options.indexOf('--target-env');
        if (index !== -1) {
            options.splice(index, 2, `--target-env=${options[index + 1]}`);
        }
        return options;
    }

    // Some tools (ex. spirv-val) needs this as a two argument, so unmerge with "="
    unmergeSpirvTargetEnv(options: string[]) {
        for (let i = 0; i < options.length; i++) {
            if (options[i].indexOf('--target-env=') === 0) {
                const parts = options[i].split('=');
                options.splice(i, 1, parts[0], parts[1]);
                break;
            }
        }
        return options;
    }

    getSpirvTargetEnv(options: string[]) {
        const index = options.indexOf('--target-env');
        if (index !== -1) {
            return [options[index], options[index + 1]];
        }

        for (const i in options) {
            if (options[i].indexOf('--target-env=') === 0) {
                return [options[i]];
            }
        }

        return []; // no target found, use tool's default
    }

    // Most flows follow the same flow:
    // 1. Assemble from spirv disassembly to a spirv binary
    // 2. Run the tool (which will dump out a binary)
    //   a. Most tools let you set the input and out file to the same binary file
    // 3. Disassemble back to disassembly
    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        const sourceDir = path.dirname(inputFilename);
        const spvBinFilename = this.getPrimaryOutputFilename(sourceDir, this.outputFilebase);

        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        execOptions.customCwd = sourceDir;

        let assemblerFlags = [inputFilename, '-o', spvBinFilename];
        assemblerFlags = assemblerFlags.concat(this.getSpirvTargetEnv(options));
        // Will fail if input SPIR-V is so bad assembler can't understand, so should let user know
        let spvasmOutput = await this.exec(this.assemblerPath, assemblerFlags, execOptions);
        let result = this.transformToCompilationResult(spvasmOutput, inputFilename);
        if (spvasmOutput.code !== 0 || !(await utils.fileExists(spvBinFilename))) {
            return result;
        }

        const spvasmFilename = path.join(sourceDir, this.outputFilebase + '.spvasm');
        // needs to update options depending on the tool
        const isValidator = compiler.endsWith('spirv-val');
        const isNonSprivOutput = compiler.endsWith('spirv-cross') || compiler.endsWith('spirv-reflect');
        if (isValidator) {
            // there is no output file, so remove what we added in optionsForFilter
            options = options.splice(2);
            options = this.unmergeSpirvTargetEnv(options);
        } else if (isNonSprivOutput) {
            options = options.splice(2);
            options.push('--output', spvasmFilename);
        } else {
            options = this.mergeSpirvTargetEnv(options);
        }

        // have tools input a binary and output it to same binary temp file
        // Unless we don't want to run spirv-dis, we still save to a .spvasm because we don't know the compiler
        // at getOutputFilename() and can just adjust the parsing in processAsm()
        for (const i in options) {
            if (options[i] === inputFilename) {
                options[i] = spvBinFilename;
                break;
            }
        }

        const spvBin = await this.exec(compiler, options, execOptions);
        result = this.transformToCompilationResult(spvBin, inputFilename);

        if (isValidator) {
            result.validatorTool = true;
        }

        if (spvBin.code !== 0 || !(await utils.fileExists(spvBinFilename)) || isValidator || isNonSprivOutput) {
            return result;
        }

        const disassemblerFlags = [spvBinFilename, '-o', spvasmFilename, '--comment'];

        // Will likely never fail
        spvasmOutput = await this.exec(this.disassemblerPath, disassemblerFlags, execOptions);
        if (spvasmOutput.code !== 0) {
            logger.error('spirv-dis failed to disassemble binary', spvasmOutput);
        }

        result = this.transformToCompilationResult(spvasmOutput, spvBinFilename);
        return result;
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        if (result.asm.startsWith('; SPIR-V')) {
            return this.spirvAsm.processAsm(result.asm, filters);
        }
        // If not SPIR-V, just display as plain text to be safe
        return super.processAsm(result, filters, options);
    }
}
