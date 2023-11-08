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

import _ from 'underscore';

import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {SPIRVAsmParser} from '../parsers/asm-parser-spirv.js';

export class HLSLCompiler extends BaseCompiler {
    protected spirvAsm: SPIRVAsmParser;
    static get key() {
        return 'hlsl';
    }

    constructor(info: any, env: any) {
        super(info, env);

        this.compiler.supportsIntel = false;
        this.spirvAsm = new SPIRVAsmParser(this.compilerProps);

        this.compiler.supportsLLVMOptPipelineView = true;
        this.compiler.llvmOptArg = ['-print-before-all', '-print-after-all'];
        this.compiler.llvmOptNoDiscardValueNamesArg = [];
    }

    override async generateAST(inputFilename, options) {
        // These options make DXC produce an AST dump
        const newOptions = _.filter(options, option => option !== '-Zi' && option !== '-Qembed_debug').concat([
            '-ast-dump',
        ]);

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.llvmAst.processAst(
            await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions),
        );
    }

    override couldSupportASTDump(version: string) {
        return version.includes('libdxcompiler');
    }

    /* eslint-disable no-unused-vars */
    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        return [
            '-Zi', // Embed debug information to get DXIL line associations
            '-Qembed_debug', // Silences the warning associated with embedded debug information
            `-Fc ${outputFilename}`, // Output object
        ];
    }
    /* eslint-enable no-unused-vars */

    override filterUserOptions(userOptions: any) {
        // RGA supports a non-standard flag --asic [ASIC] which must be removed when compiling with DXC
        const options = userOptions.slice(0);
        // Scan for the RGA-specific argument --asic and strip it and its corresponding argument
        // Assumes the argument exists at most once (compilation will fail if supplied more than
        // once regardless)
        for (let i = 0; i !== options.length; ++i) {
            const option = options[i];
            if (option === '--asic') {
                options.splice(i, 2);
                break;
            }
        }
        return options;
    }

    override async processAsm(result, filters, options) {
        if (this.isSpirv(result.asm)) {
            return this.spirvAsm.processAsm(result.asm, filters);
        }

        return super.processAsm(result, filters, options);
    }

    isSpirv(code) {
        return code.startsWith('; SPIR-V');
    }
}
