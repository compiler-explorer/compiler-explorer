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

import {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
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

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = ['--target=avx2-i32x8', '-g'];
        if (filters.binary || filters.binaryObject) {
            // TODO(#7272) this is a little hacky but it's a way to get the output to be a `.o` file that we can then
            //  link with the executableLinker later in the `.s` file (even though that's also a bad extension really).
            //  For binaryObject, we then also have to rename it to `.s` later to match the getOutputFilename() name,
            //  but if we leave it as `.o` then `ispc` complains about the "Emitting object file, but the filename has
            //  suffix '.s'".
            options = options.concat('-o', this.filename(utils.changeExtension(outputFilename, '.o')));
        } else {
            options = options.concat('--emit-asm', '-o', this.filename(outputFilename));
        }
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
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

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const result = await super.runCompiler(compiler, options, inputFilename, execOptions, filters);
        if (result.code !== 0) return result;
        // Rely on the fact we definitely put a `-o` in the options with the actual filename.
        const inFile = options[options.indexOf('-o') + 1];
        const outFile = utils.changeExtension(inFile, '.s');
        if (filters?.binary) {
            const linkResult = await this.exec(this.executableLinker, ['-o', outFile, inFile], execOptions);
            if (linkResult.code !== 0) {
                return {...result, ...this.transformToCompilationResult(linkResult, inputFilename)};
            }
        } else if (filters?.binaryObject) {
            // We outputted as `.o` so rename to `.s` to match the getOutputFilename() name to avoid ispc's moaning.
            await fs.rename(inFile, outFile);
        }
        return result;
    }

    override buildExecutable(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        return this.runCompiler(compiler, options, inputFilename, execOptions, {binary: true});
    }
}
