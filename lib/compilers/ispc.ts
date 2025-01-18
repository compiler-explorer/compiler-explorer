// Copyright (c) 2017, Compiler Explorer Authors
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

import fs from 'fs-extra';
import Semver from 'semver';

import {
    CacheKey,
    CompilationCacheKey,
    CompilationResult,
    ExecutionOptionsWithEnv,
} from '../../types/compilation/compilation.interfaces.js';
import {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {asSafeVer} from '../utils.js';
import * as utils from '../utils.js';

import {ISPCParser} from './argument-parsers.js';

export class ISPCCompiler extends BaseCompiler {
    private readonly executableLinker: string;

    static get key() {
        return 'ispc';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit-llvm-text'];
        // TODO(#7150) do away with this kind of thing and share the same linker for everything.
        this.executableLinker = this.compilerProps<string>(`compiler.${this.compiler.id}.executableLinker`);
    }

    override couldSupportASTDump(version: string) {
        return Semver.gte(asSafeVer(this.compiler.semver), '1.18.0', true);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey): string {
        const outputFilename = super.getOutputFilename(dirPath, outputFilebase, key);
        if (this.isCacheKey(key)) {
            // ispc gets a bit annoyed about the output filename extension. If we're in binary object mode we _have_ to
            // output a `.o` file, else ispc complains.
            if (key.filters.binary || key.filters.binaryObject) {
                return utils.changeExtension(outputFilename, '.o');
            }
        }
        return outputFilename;
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey) {
        // ispc gets a bit annoyed about the output filename extension. If we're in binary object mode we _have_ to
        // output a `.o` file, else ispc complains. We can't defer this to getOutputFilename, else we'll rename the
        // output of executed but non-binary, non-binary-object files, which need to remain `.s`.
        return utils.changeExtension(super.getOutputFilename(dirPath, outputFilebase, key), '.o');
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options = ['--target=avx2-i32x8', '-g', '-o', this.filename(outputFilename)];
        if (!filters.binary && !filters.binaryObject) {
            options.push('--emit-asm');
        }
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options.push(...this.compiler.intelAsm.split(' '));
        }
        return options;
    }

    override async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const newOptions = [
            ...options,
            ...unwrap(this.compiler.irArg),
            '-o',
            this.getIrOutputFilename(inputFilename, filters),
        ];
        return super.generateIR(inputFilename, newOptions, irOptions, produceCfg, filters);
    }

    override getArgumentParserClass() {
        return ISPCParser;
    }

    override async generateAST(inputFilename: string, options: string[]): Promise<ResultLine[]> {
        // These options make Clang produce an AST dump
        const newOptions = options.filter(option => option !== '--colored-output').concat(['--ast-dump']);

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.llvmAst.processAst(
            await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions),
        );
    }

    override getLibLinkInfo(
        filters: ParseFiltersAndOutputOptions,
        libraries: SelectedLibraryVersion[],
        toolchainPath: string,
        dirPath: string,
    ) {
        // Prevent any library linking flags from being passed to ispc during compilation.
        return {libLinks: [], libPathsAsFlags: [], staticLibLinks: []};
    }

    override isCfgCompiler() {
        return true;
    }

    async linkExecutable(options: string[], execOptions: ExecutionOptionsWithEnv) {
        // Rely on the fact we definitely put a `-o` in the options.
        const outputFile = options[options.indexOf('-o') + 1];
        const renamedFile = outputFile + '.tmp';
        await fs.rename(outputFile, renamedFile);
        return await this.exec(this.executableLinker, ['-o', outputFile, renamedFile], execOptions);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const result = await super.runCompiler(compiler, options, inputFilename, execOptions, filters);
        if (result.code !== 0) return result;
        if (filters?.binary) {
            const linkResult = await this.linkExecutable(options, execOptions);
            if (linkResult.code !== 0) {
                return {...result, ...this.transformToCompilationResult(linkResult, inputFilename)};
            }
        }
        return result;
    }
}
