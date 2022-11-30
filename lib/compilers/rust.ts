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

import {BasicExecutionResult, UnprocessedExecResult} from '../../types/execution/execution.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {BuildEnvDownloadInfo} from '../buildenvsetup/buildenv.interfaces';
import {parseRustOutput} from '../utils';

import {RustParser} from './argument-parsers';

export class RustCompiler extends BaseCompiler {
    linker: string;

    static get key() {
        return 'rust';
    }

    constructor(info, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
        this.compiler.supportsIrView = true;
        this.compiler.supportsLLVMOptPipelineView = true;
        this.compiler.supportsRustMirView = true;

        const isNightly = info.name === 'nightly' || info.semver === 'nightly';
        // Macro expansion (-Zunpretty=expanded) and HIR (-Zunpretty=hir-tree)
        // are only available for Nightly
        this.compiler.supportsRustMacroExpView = isNightly;
        this.compiler.supportsRustHirView = isNightly;

        this.compiler.irArg = ['--emit', 'llvm-ir'];
        this.compiler.llvmOptArg = ['-C', 'llvm-args=-print-after-all -print-before-all'];
        this.compiler.llvmOptModuleScopeArg = ['-C', 'llvm-args=-print-module-scope'];
        this.compiler.llvmOptNoDiscardValueNamesArg = isNightly ? ['-Z', 'fewer-names=no'] : [];
        this.linker = this.compilerProps<string>('linker');
    }

    override getSharedLibraryPathsAsArguments(libraries, libDownloadPath) {
        return [];
    }

    override getSharedLibraryLinks(libraries): string[] {
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
        options,
        inputFilename,
        libIncludes,
        libOptions,
        libPaths,
        libLinks,
        userOptions,
        staticLibLinks,
    ) {
        return options.concat(userOptions, libIncludes, libOptions, libPaths, libLinks, staticLibLinks, [
            this.filename(inputFilename),
        ]);
    }

    override async setupBuildEnvironment(key, dirPath): Promise<BuildEnvDownloadInfo[]> {
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

    override optionsForBackend(backendOptions, outputFilename) {
        // The super class handles the GCC dump files that may be needed by
        // rustc-cg-gcc subclass.
        const opts = super.optionsForBackend(backendOptions, outputFilename);

        if (backendOptions.produceRustMir && this.compiler.supportsRustMirView) {
            const of = this.getRustMirOutputFilename(outputFilename);
            opts.push('--emit', `mir=${of}`);
        }
        return opts;
    }

    override optionsForFilter(filters, outputFilename, userOptions) {
        let options = ['-C', 'debuginfo=1', '-o', this.filename(outputFilename)];

        const userRequestedEmit = _.any(userOptions, opt => opt.includes('--emit'));
        if (filters.binary) {
            options = options.concat(['--crate-type', 'bin']);
            if (this.linker) {
                options = options.concat(`-Clinker=${this.linker}`);
            }
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
