// Copyright (c) 2018, Adrian Bibby Walther
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

import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import {CompilerOverrideType, type ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler, c_default_toolchain_description, c_value_placeholder} from '../base-compiler.js';
import {BinaryInfoLinux} from '../binaries/binary-utils.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {CurrentHostExecHelper} from '../execution/execution-triple.js';
import {
    clang_style_sysroot_flag,
    clang_style_toolchain_flag,
    getSysrootByToolchainPath,
    getToolchainPathWithOptionsArr,
    removeSysrootArg,
    removeToolchainArg,
} from '../toolchain-utils.js';
import {LLCParser} from './argument-parsers.js';

export class LLCCompiler extends BaseCompiler {
    static get key() {
        return 'llc';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIntel = true;
        this.compiler.optPipeline = {
            arg: ['-print-after-all', '-print-before-all'],
            moduleScopeArg: ['-print-module-scope'],
            noDiscardValueNamesArg: [],
        };
        // llc doesn't link, so the toolchain can't come from its options like it does for clang
        this.toolchainPath = this.compilerProps<string>(
            `compiler.${this.compiler.id}.linkerToolchain`,
            this.compilerProps<string>('linkerToolchain', '') || undefined,
        );
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = ['-o', this.filename(outputFilename)];
        if (filters.intel && !filters.binary && !filters.binaryObject)
            options = options.concat('-x86-asm-syntax=intel');
        if (filters.binary || filters.binaryObject) options = options.concat('-filetype=obj');
        return options;
    }

    override getTargetFlags(): string[] {
        return [`-mtriple=${c_value_placeholder}`];
    }

    override getAllPossibleTargetFlags(): string[][] {
        return [this.getTargetFlags()];
    }

    override async populatePossibleOverrides() {
        await super.populatePossibleOverrides();

        const possibleToolchains = await this.getPossibleToolchains();
        if (possibleToolchains.length > 0) {
            this.compiler.possibleOverrides?.push({
                type: 'options',
                name: CompilerOverrideType.toolchain,
                display_title: 'Toolchain',
                description: c_default_toolchain_description,
                // Taken back out of the options in runCompiler, it's only used for linking
                flags: [clang_style_toolchain_flag + c_value_placeholder],
                values: possibleToolchains,
            });
        }
    }

    override changeOptionsBasedOnOverrides(options: string[], overrides: ConfiguredOverrides): string[] {
        return removeSysrootArg(super.changeOptionsBasedOnOverrides(options, overrides));
    }

    getLinkerArguments(toolchainPath: string | undefined): string[] {
        if (!toolchainPath) return [];

        const args = [clang_style_toolchain_flag + toolchainPath];
        // A cross toolchain lives in a folder named after its target triple, with the sysroot inside it
        const sysroot = getSysrootByToolchainPath(toolchainPath);
        if (sysroot) args.push(`--target=${path.basename(toolchainPath)}`, clang_style_sysroot_flag + sysroot);
        return args;
    }

    async linkExecutable(
        options: string[],
        execOptions: ExecutionOptionsWithEnv,
        toolchainPath: string | undefined,
    ): Promise<UnprocessedExecResult | undefined> {
        const outputFile = options[options.indexOf('-o') + 1];
        const isCrossToolchain = !!toolchainPath && !!getSysrootByToolchainPath(toolchainPath);
        if (!isCrossToolchain) {
            // Without a cross toolchain all we can link for is the host
            const info = await BinaryInfoLinux.readFile(outputFile);
            const hostInstructionSet = CurrentHostExecHelper.getInstructionSetByNodeJSArch(os.arch());
            if (info?.instructionSet !== hostInstructionSet) return undefined;
        }

        const objectFile = outputFile + '.o';
        await fs.rename(outputFile, objectFile);
        const clang = path.join(path.dirname(this.compiler.exe), 'clang');
        // llc defaults to the static relocation model, which can't be linked into a PIE.
        return await this.exec(
            clang,
            [...this.getLinkerArguments(toolchainPath), '-no-pie', '-o', outputFile, objectFile],
            execOptions,
        );
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const toolchainPath = getToolchainPathWithOptionsArr(null, options) ?? this.toolchainPath;
        const llcOptions = removeToolchainArg(options);
        const result = await super.runCompiler(compiler, llcOptions, inputFilename, execOptions, filters);
        if (result.code !== 0 || !filters?.binary) return result;
        const linkResult = await this.linkExecutable(llcOptions, execOptions, toolchainPath);
        if (!linkResult && filters.execute) {
            return {
                ...result,
                code: 1,
                stderr: [{text: 'Select a toolchain for this target architecture to link an executable'}],
            };
        }
        if (linkResult && linkResult.code !== 0) {
            return {...result, ...this.transformToCompilationResult(linkResult, inputFilename)};
        }
        return result;
    }

    override getArgumentParserClass() {
        return LLCParser;
    }
}
