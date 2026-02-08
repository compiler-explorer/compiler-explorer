// Copyright (c) 2026, Compiler Explorer Authors
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

import path from 'node:path';

import type {ExecutionOptionsWithEnv, FiledataPair} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class TinyGoCompiler extends BaseCompiler {
    private readonly tinygoRoot: string;
    private readonly goRoot: string | undefined;

    static get key() {
        return 'tinygo';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.tinygoRoot = path.dirname(path.dirname(compilerInfo.exe));

        const group = this.compiler.group;
        this.goRoot = this.compilerProps<string | undefined>(
            'goroot',
            this.compilerProps<string | undefined>(`group.${group}.goroot`),
        );
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['build', '-o', outputFilename];
    }

    override getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const options = super.getDefaultExecOptions();

        options.env = {
            ...options.env,
            TINYGOROOT: this.tinygoRoot,
        };

        if (this.goRoot) {
            options.env.GOROOT = this.goRoot;
            const goBin = path.join(this.goRoot, 'bin');
            options.env.PATH = options.env.PATH ? `${goBin}:${options.env.PATH}` : goBin;
        }

        return options;
    }

    override fixFiltersBeforeCacheKey(filters: ParseFiltersAndOutputOptions, options: string[], files: FiledataPair[]) {
        // TinyGo is LLVM-based and always produces a binary — force binary mode
        // to avoid the base compiler adding -S (which TinyGo doesn't understand).
        filters.binary = true;
        super.fixFiltersBeforeCacheKey(filters, options, files);
    }

    override isCfgCompiler() {
        return true;
    }
}
