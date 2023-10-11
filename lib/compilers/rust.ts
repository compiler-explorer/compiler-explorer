// Copyright (c) 2016, Compiler Explorer Authors
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

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {BasicExecutionResult, UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import type {BuildEnvDownloadInfo} from '../buildenvsetup/buildenv.interfaces.js';
import {parseRustOutput} from '../utils.js';

import {RustParser} from './argument-parsers.js';
import {CompilerOverrideType} from '../../types/compilation/compiler-overrides.interfaces.js';
import {SemVer} from 'semver';

export class RustCompiler extends BaseCompiler {
    linker: string;

    static get key() {
        return 'rust';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
        this.compiler.supportsIrView = true;
        this.compiler.supportsLLVMOptPipelineView = true;
        this.compiler.supportsRustMirView = true;

        const isNightly = this.isNightly();
        // Macro expansion (-Zunpretty=expanded) and HIR (-Zunpretty=hir-tree)
        // are only available for Nightly
        this.compiler.supportsRustMacroExpView = isNightly;
        this.compiler.supportsRustHirView = isNightly;

        this.compiler.irArg = ['--emit', 'llvm-ir'];
        this.compiler.minIrArgs = ['--emit=llvm-ir'];
        this.compiler.llvmOptArg = ['-C', 'llvm-args=-print-after-all -print-before-all'];
        this.compiler.llvmOptModuleScopeArg = ['-C', 'llvm-args=-print-module-scope'];
        this.compiler.llvmOptNoDiscardValueNamesArg = isNightly ? ['-Z', 'fewer-names=no'] : [];
        this.linker = this.compilerProps<string>('linker');
    }

    private isNightly() {
        return (
            this.compiler.name === 'nightly' ||
            this.compiler.semver === 'nightly' ||
            this.compiler.semver === 'beta' ||
            this.compiler.semver.includes('master') ||
            this.compiler.semver.includes('trunk')
        );
    }

    override async populatePossibleOverrides() {
        const possibleEditions = await RustParser.getPossibleEditions(this);
        if (possibleEditions.length > 0) {
            let defaultEdition: undefined | string = undefined;
            if (!this.compiler.semver || this.isNightly()) {
                defaultEdition = '2021';
            } else {
                const compilerVersion = new SemVer(this.compiler.semver);
                if (compilerVersion.compare('1.56.0') >= 0) {
                    defaultEdition = '2021';
                }
            }

            this.compiler.possibleOverrides?.push({
                name: CompilerOverrideType.edition,
                display_title: 'Edition',
                description:
                    'The default edition for Rust compilers is usually 2015. ' +
                    'Some editions might not be available for older compilers.',
                flags: ['--edition', '<value>'],
                values: possibleEditions.map(ed => {
                    return {name: ed, value: ed};
                }),
                default: defaultEdition,
            });
        }

        await super.populatePossibleOverrides();
    }

    override getSharedLibraryPathsAsArguments(libraries, libDownloadPath) {
        return [];
    }

    override getSharedLibraryLinks(libraries: any[]): string[] {
        return [];
    }

    override getIncludeArguments(libraries) {
        const includeFlag = '--extern';
        return libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion || !foundVersion.name) return [];
            const lowercaseLibName = foundVersion.name.replaceAll('-', '_');
            return foundVersion.path.flatMap(rlib => {
                return [
                    includeFlag,
                    `${lowercaseLibName}=${foundVersion.name}/build/debug/${rlib}`,
                    '-L',
                    `dependency=${foundVersion.name}/build/debug/deps`,
                ];
            });
        });
    }

    override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        return options.concat(userOptions, libIncludes, libOptions, libPaths, libLinks, staticLibLinks, [
            this.filename(inputFilename),
        ]);
    }

    override async setupBuildEnvironment(key: any, dirPath: string): Promise<BuildEnvDownloadInfo[]> {
        if (this.buildenvsetup) {
            const libraryDetails = await this.getRequiredLibraryVersions(key.libraries);
            return this.buildenvsetup.setup(key, dirPath, libraryDetails);
        } else {
            return [];
        }
    }

    override fixIncompatibleOptions(options: string[], userOptions: string[]): string[] {
        if (userOptions.filter(option => option.startsWith('--color=')).length > 0) {
            options = options.filter(option => !option.startsWith('--color='));
        }
        return options;
    }

    override optionsForBackend(backendOptions: Record<string, any>, outputFilename: string) {
        // The super class handles the GCC dump files that may be needed by
        // rustc-cg-gcc subclass.
        const opts = super.optionsForBackend(backendOptions, outputFilename);

        if (backendOptions.produceRustMir && this.compiler.supportsRustMirView) {
            const of = this.getRustMirOutputFilename(outputFilename);
            opts.push('--emit', `mir=${of}`);
        }
        return opts;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        let options = ['-C', 'debuginfo=1', '-o', this.filename(outputFilename)];

        const userRequestedEmit = _.any(unwrap(userOptions), opt => opt.includes('--emit'));
        if (filters.binary) {
            options = options.concat(['--crate-type', 'bin']);
            if (this.linker) {
                options = options.concat(`-Clinker=${this.linker}`);
            }
        } else if (filters.binaryObject) {
            options = options.concat(['--crate-type', 'lib']);
        } else {
            if (!userRequestedEmit) {
                options = options.concat('--emit', 'asm');
            }
            if (filters.intel) options = options.concat('-Cllvm-args=--x86-asm-syntax=intel');
            options = options.concat(['--crate-type', 'rlib']);
        }
        return options;
    }

    // Override the IR file name method for rustc because the output file is different from clang.
    override getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        const outputFilename = this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase);
        // As per #4054, if we are asked for binary mode, the output will be in the .s file, no .ll will be emited
        if (!filters.binary) {
            return outputFilename.replace('.s', '.ll');
        }
        return outputFilename;
    }

    override getArgumentParser() {
        return RustParser;
    }

    override isCfgCompiler(/*compilerVersion*/) {
        return true;
    }

    override processExecutionResult(input: UnprocessedExecResult, inputFilename?: string): BasicExecutionResult {
        return {
            ...input,
            stdout: parseRustOutput(input.stdout, inputFilename),
            stderr: parseRustOutput(input.stderr, inputFilename),
        };
    }
}
