// Copyright (c) 2024, Compiler Explorer Authors
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

import {BaseCompiler, c_value_placeholder} from '../base-compiler.js';
import {KotlinNativeParser} from './argument-parsers.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import path from 'path';
import {filterUserOptionsWithArg} from '../utils.js';

const ARTIFACTS_DIR: string = 'artifacts';

export class KotlinNativeCompiler extends BaseCompiler {
    static get key() {
        return 'kotlin-native';
    }

    javaHome: string;
    phaseToDumpLlvmIrAfter: string;
    supportedTargets: string[] = [];

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        this.javaHome = this.compilerProps<string>(`compiler.${this.compiler.id}.java_home`);

        this.compiler.supportsIrView = true;
        this.phaseToDumpLlvmIrAfter = 'LTOBitcodeOptimization';
        const skipCompilerPhases = ['ObjectFiles', 'Linker'];
        this.compiler.irArg = [
            `-Xsave-llvm-ir-after=${this.phaseToDumpLlvmIrAfter}`,
            `-Xsave-llvm-ir-directory=${ARTIFACTS_DIR}`,
            `-Xdisable-phases=${skipCompilerPhases.join(',')}`,
        ];
    }

    override getTargetFlags(): string[] {
        return ['-target', c_value_placeholder];
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.javaHome) {
            execOptions.env.JAVA_HOME = this.javaHome;
        }
        return execOptions;
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string): string {
        return path.join(dirPath, outputFilebase + '.kexe');
    }

    override getOutputFilename(dirPath: string): string {
        // This is actually an assembly file, despite the .o extension.
        // This is because the Kotlin compiler always passes this file name to Clang's `-o` option,
        // but we insert `-S`, so the assembly text is written to this .o file instead.
        return path.join(dirPath, ARTIFACTS_DIR, `${this.outputFilebase}.kexe.o`);
    }

    override getObjdumpOutputFilename(defaultOutputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        if (filters.binaryObject) {
            return defaultOutputFilename;
        }
        return this.getExecutableFilename(path.dirname(path.dirname(defaultOutputFilename)), this.outputFilebase);
    }

    override getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        // This file is the result of the -Xsave-llvm-ir-after=Codegen CLI option.
        return path.join(path.dirname(inputFilename), ARTIFACTS_DIR, `out.${this.phaseToDumpLlvmIrAfter}.ll`);
    }

    protected override getArgumentParser(): any {
        return KotlinNativeParser;
    }

    private konanProperty(key: string, target: string, value: string): string {
        return `${key}.${target}=${value}`;
    }

    /**
     * The Kotlin/Native compiler doesn't have an option for emitting assembly, but it invokes
     * Clang under the hood and allows to customize the flags passed to it, so we take that opportunity
     * to tell Clang to emit an assembly file by passing `-S`. All the other flags are just what the Kotlin compiler
     * passes to Clang by default.
     */
    private overrideKonanProperties(filters: ParseFiltersAndOutputOptions): string {
        let clangFlags = ['-cc1', '-disable-llvm-passes', '-x', 'ir']; // This is the default set of flags that Kotlin passes to Clang

        if (!filters.binary && !filters.binaryObject) {
            clangFlags.push('-S');
            if (this.compiler.intelAsm && filters.intel) {
                clangFlags = clangFlags.concat(this.compiler.intelAsm.split(' '));
            }
        }

        // The Kotlin/Native compiler allows to override Clang flags for specific targets only, not for all targets at
        // once.
        return this.supportedTargets
            .map(target => this.konanProperty('clangFlags', target, clangFlags.join(' ')))
            .join(';');
    }

    override filterUserOptions(userOptions: string[]): string[] {
        const forbiddenFlags = new Set(['-nomain', '-nopack', '-script']);

        const oneArgForbiddenList = new Set(['-o', '--output', '-p', '--produce', '-kotlin-home']);

        // filter options without extra arguments
        userOptions = (userOptions || []).filter(option => !forbiddenFlags.has(option));

        return filterUserOptionsWithArg(userOptions, oneArgForbiddenList);
    }

    protected override optionsForFilter(filters: ParseFiltersAndOutputOptions): string[] {
        const options = [
            '-produce',
            'program',
            '-g',
            `-Xtemporary-files-dir=${ARTIFACTS_DIR}`,
            '-o',
            this.outputFilebase,
        ];

        // This option enables compiler caches.
        // Compiler caches allow us to reuse already compiled libraries when linking the final executable.
        // It's very useful in CE, since it makes the resulting assembly file much smaller and the compilation process
        // much faster.
        // If we didn't use compiler caches, we'd get the whole compiled stdlib in our assembly file — hundreds of
        // thousands lines of code.
        // It's somewhat weird: it expects a path to a directory with depended-on libraries, but stdlib is
        // an exception: it doesn't have to be in that path, the compiler will use the stdlib from its own distribution.
        // However, the compiler will not use any caches if we don't pass it this flag.
        // Basically, since for the purposes of Compiler Explorer we only need stdlib (for now), we can pass
        // an arbitrary path with this option, as long as we pass the option itself.
        options.push('-Xauto-cache-from=');

        if (!filters.binary) {
            if (!filters.binaryObject) {
                // Add Clang flags to emit assembly
                const overriddenProperties = this.overrideKonanProperties(filters);
                options.push(`-Xoverride-konan-properties=${overriddenProperties}`);
            }
            options.push('-Xdisable-phases=Linker');
        }

        return options;
    }

    protected override getSharedLibraryPathsAsArguments() {
        return []; // Not applicable to this compiler
    }
}
