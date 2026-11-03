// Copyright (c) 2025, Compiler Explorer Authors
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

import semver from 'semver';

import type {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler, c_value_placeholder} from '../base-compiler.js';
import type {CompilationEnvironment} from '../compilation-env.js';
import {MojoParser} from './argument-parsers.js';

export class MojoCompiler extends BaseCompiler {
    static get key() {
        return 'mojo';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit', 'llvm'];
        this.compiler.minIrArgs = ['--emit=llvm'];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return [
            'build',
            ...super.optionsForFilter(filters, outputFilename).map(option => {
                if (option === '-S') return '--emit=asm';
                // Full debug info changes generated code; line tables suffice for source mapping.
                if (option === '-g') return '-g1';
                return option;
            }),
        ];
    }

    override getSharedLibraryPathsAsArguments(
        libraries: SelectedLibraryVersion[],
        libDownloadPath: string | undefined,
        toolchainPath: string | undefined,
        dirPath: string,
    ) {
        const version = semver.coerce(this.compiler.version);
        // Older Mojo releases reject -Xlinker.
        if (!version || semver.lt(version, '0.26.1')) {
            return [];
        }

        return super.getSharedLibraryPathsAsArguments(libraries, libDownloadPath, toolchainPath, dirPath);
    }

    override async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const newOptions = options.filter(
            (opt, idx, arr) =>
                opt !== '-o' &&
                opt !== '--emit' &&
                !opt.startsWith('--emit=') &&
                arr[idx - 1] !== '-o' &&
                arr[idx - 1] !== '--emit',
        );
        newOptions.push('-o', this.getIrOutputFilename(inputFilename));
        return await super.generateIR(inputFilename, newOptions, irOptions, produceCfg, filters);
    }

    override getArgumentParserClass() {
        return MojoParser;
    }

    override getAllPossibleTargetFlags(): string[][] {
        return [[`--target-triple=${c_value_placeholder}`], ['--target-triple', c_value_placeholder]];
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        execOptions.env.MODULAR_CRASH_REPORTING_ENABLED = 'false';
        return execOptions;
    }
}
