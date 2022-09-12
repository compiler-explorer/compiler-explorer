// Copyright (c) 2022, Compiler Explorer Authors
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

import _ from 'underscore';

import {BaseCompiler} from '../base-compiler';
import {logger} from '../logger';
import {SPIRVAsmParser} from '../parsers/asm-parser-spirv';
import * as utils from '../utils';

export class CLSPVCompiler extends BaseCompiler {
    disassemblerPath: any;

    static get key() {
        return 'clspv';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.asm = new SPIRVAsmParser();

        this.disassemblerPath = this.compilerProps('disassemblerPath');
    }

    override prepareArguments(userOptions, filters, backendOptions, inputFilename, outputFilename, libraries) {
        let options = this.optionsForFilter(filters, outputFilename);
        backendOptions = backendOptions || {};

        if (this.compiler.options) {
            const compilerOptions = _.filter(
                utils.splitArguments(this.compiler.options),
                option => option !== '-fno-crash-diagnostics',
            );

            options = options.concat(compilerOptions);
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(this.compiler.optArg);
        }

        const libIncludes = this.getIncludeArguments(libraries);
        const libOptions = this.getLibraryOptions(libraries);
        let libLinks: any = [];
        let libPaths: any = [];
        let staticLibLinks: any = [];

        if (filters.binary) {
            libLinks = this.getSharedLibraryLinks(libraries);
            libPaths = this.getSharedLibraryPathsAsArguments(libraries);
            staticLibLinks = this.getStaticLibraryLinks(libraries);
        }

        userOptions = this.filterUserOptions(userOptions) || [];
        return options.concat(
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            userOptions,
            [this.filename(inputFilename)],
            staticLibLinks,
        );
    }

    override optionsForFilter(filters, outputFilename) {
        const sourceDir = path.dirname(outputFilename);
        const spvBinFilename = path.join(sourceDir, this.outputFilebase + '.spv');
        return ['-o', spvBinFilename];
    }

    getPrimaryOutputFilename(dirPath, outputFilebase) {
        return path.join(dirPath, `${outputFilebase}.spv`);
    }

    // TODO: Check this to see if it needs key
    override getOutputFilename(dirPath, outputFilebase) {
        return path.join(dirPath, `${outputFilebase}.spvasm`);
    }

    override async runCompiler(compiler, options, inputFilename, execOptions) {
        const sourceDir = path.dirname(inputFilename);
        const spvBinFilename = path.join(sourceDir, this.outputFilebase + '.spv');

        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        execOptions.customCwd = path.dirname(inputFilename);

        const spvBin = await this.exec(compiler, options, execOptions);
        const result = this.transformToCompilationResult(spvBin, inputFilename);

        if (spvBin.code !== 0 || !(await utils.fileExists(spvBinFilename))) {
            return result;
        }

        const spvasmFilename = path.join(sourceDir, this.outputFilebase + '.spvasm');
        const disassemblerFlags = [spvBinFilename, '-o', spvasmFilename];

        const spvasmOutput = await this.exec(this.disassemblerPath, disassemblerFlags, execOptions);
        if (spvasmOutput.code !== 0) {
            logger.error('SPIR-V binary to text failed', spvasmOutput);
        }

        result.stdout = result.stdout.concat(utils.parseOutput(spvasmOutput.stdout));
        result.stderr = result.stderr.concat(utils.parseOutput(spvasmOutput.stderr));
        return result;
    }
}
