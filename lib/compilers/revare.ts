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

import type {CacheKey, CompilationCacheKey} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class RevareCompiler extends BaseCompiler {
    static get key() {
        return 'revare';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.outputFilebase = 'example';
        this.compiler.supportsIrView = true;
        this.compiler.irArg = [];
        this.compiler.supportsIntel = false;
    }

    // Revare's command line takes verbs and positional arguments only. All build
    // settings live in config.toml, not in flags. The `codegen` verb runs the
    // build up to LLVM IR and writes it to the path it is given, without running
    // opt, clang or the linker.
    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        return [];
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
        // `codegen <file> <out.ll>` writes the LLVM IR to the named path, which is what
        // getOutputFilename() below points CE at. User options are not passed: a stray one
        // would be read as a file name.
        return ['codegen', this.filename(inputFilename), `${this.outputFilebase}.ll`];
    }

    // The primary output is the emitted LLVM IR itself.
    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey): string {
        return path.join(dirPath, `${outputFilebase}.ll`);
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'llvm-ir';
    }

    override getSharedLibraryPathsAsArguments(): string[] {
        return [];
    }
}
