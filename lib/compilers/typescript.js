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

import {BaseCompiler} from '../base-compiler';

import {TypeScriptNativeParser} from './argument-parsers';

export class TypeScriptCompiler extends BaseCompiler {
    static get key() {
        return 'typescript';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.compiler.supportsIntel = false;
        this.compiler.supportsIrView = true;

        this.tscJit = this.compilerProps(`compiler.${this.compiler.id}.exe`);
        this.tscSharedLib = this.compilerProps(`compiler.${this.compiler.id}.sharedlibs`);
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    optionsForFilter(filters, outputFilename) {
        return [this.filename(outputFilename)];
    }

    async handleInterpreting(key, executeParameters) {
        executeParameters.args = [
            '--emit=jit',
            this.tscSharedLib ? '--shared-libs=' + this.tscSharedLib : '-nogc',
            ...executeParameters.args,
        ];

        return await super.handleInterpreting(key, executeParameters);
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        // These options make Clang produce an IR
        const newOptions = ['--emit=mlir-llvm', inputFilename];

        if (!this.tscSharedLib) {
            newOptions.push('-nogc');
        }

        const output = await this.runCompilerRawOutput(
            this.tscJit,
            newOptions,
            this.filename(inputFilename),
            execOptions,
        );
        if (output.code !== 0) {
            return [{text: 'Failed to run compiler to get MLIR code'}];
        }

        return {code: 0};
    }

    async generateIR(inputFilename, options, filters) {
        // These options make Clang produce an IR
        const newOptions = ['--emit=llvm', inputFilename];

        if (!this.tscSharedLib) {
            newOptions.push('-nogc');
        }

        const execOptions = this.getDefaultExecOptions();
        // TODO: maybe this isn't needed?
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const output = await this.runCompilerRawOutput(
            this.tscJit,
            newOptions,
            this.filename(inputFilename),
            execOptions,
        );
        if (output.code !== 0) {
            return [{text: 'Failed to run compiler to get IR code'}];
        }

        filters.commentOnly = false;
        filters.libraryCode = true;
        filters.directives = true;

        const ir = await this.llvmIr.process(output.stderr, filters);
        return ir.asm;
    }

    isCfgCompiler() {
        return true;
    }

    getArgumentParser() {
        return TypeScriptNativeParser;
    }
}
