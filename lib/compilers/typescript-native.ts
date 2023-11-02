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

import Semver from 'semver';
import path from 'path';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {asSafeVer, changeExtension} from '../utils.js';

import {TypeScriptNativeParser} from './argument-parsers.js';
import {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';

export class TypeScriptNativeCompiler extends BaseCompiler {
    static get key() {
        return 'typescript';
    }

    tscJit: string;
    tscSharedLib: string;
    tscNewOutput: boolean;
    tscAsmOutput: boolean;

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);

        this.tscJit = this.compiler.exe;
        this.tscSharedLib = this.compilerProps<string>(`compiler.${this.compiler.id}.sharedlibs`);
        this.tscNewOutput = Semver.gt(asSafeVer(this.compiler.semver || '0.0.0'), '0.0.32', true);
        this.tscAsmOutput = Semver.gt(asSafeVer(this.compiler.semver || '0.0.0'), '0.0.34', true);

        this.compiler.irArg = ['--emit=llvm'];
        this.compiler.supportsIntel = this.tscAsmOutput;
        this.compiler.supportsIrView = true;
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        return [];
    }

    override optionsForBackend(backendOptions: Record<string, any>, outputFilename: string): string[] {
        const addOpts: string[] = [];

        addOpts.push(this.tscAsmOutput ? '--emit=asm' : '--emit=mlir');

        if (!this.tscSharedLib) {
            addOpts.push('-nogc');
        }

        if (this.tscNewOutput) {
            addOpts.push('-o=' + this.filename(outputFilename));
        }

        return addOpts;
    }

    override getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        const outputFilename = this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase);
        // As per #4054, if we are asked for binary mode, the output will be in the .s file, no .ll will be emited
        if (!filters.binary) {
            return changeExtension(outputFilename, '.ll');
        }
        return outputFilename;
    }

    override async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const newOptions = [...options.filter(e => !e.startsWith('--emit=') && !e.startsWith('-o='))];
        if (this.tscNewOutput) {
            newOptions.push('-o=' + this.getIrOutputFilename(inputFilename, filters));
        }

        return await super.generateIR(inputFilename, newOptions, irOptions, produceCfg, filters);
    }

    override async processIrOutput(output, irOptions: LLVMIrBackendOptions, filters: ParseFiltersAndOutputOptions) {
        if (this.tscNewOutput) {
            return await super.processIrOutput(output, irOptions, filters);
        }

        return this.llvmIr.process(output.stderr.map(l => l.text).join('\n'), irOptions);
    }

    override async handleInterpreting(key, executeParameters: ExecutableExecutionOptions) {
        executeParameters.args = [
            '--emit=jit',
            this.tscSharedLib ? '--shared-libs=' + this.tscSharedLib : '-nogc',
            ...executeParameters.args,
        ];

        return await super.handleInterpreting(key, executeParameters);
    }

    override isCfgCompiler() {
        return true;
    }

    override getArgumentParser() {
        return TypeScriptNativeParser;
    }
}
