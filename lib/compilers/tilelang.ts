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

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {resolvePathFromAppRoot} from '../utils.js';
import {BaseParser} from './argument-parsers.js';

export class TileLangCompiler extends BaseCompiler {
    private compilerWrapperPath: string;

    static get key() {
        return 'tilelang';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper', '') ||
            resolvePathFromAppRoot('etc', 'scripts', 'tilelang_wrapper.py');
    }

    override getCompilerResultLanguageId(_filters?: ParseFiltersAndOutputOptions): string | undefined {
        // Default compile-only target is CPU C (D7). CUDA-in-CE is later.
        return 'nc';
    }

    override optionsForFilter(_filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        // Wrapper contract: python3 -I tilelang_wrapper.py --output_file out.s example.py
        return ['-I', this.compilerWrapperPath, '--output_file', outputFilename];
    }

    override getArgumentParserClass() {
        return BaseParser;
    }

    override getDefaultFilters() {
        return {
            intel: false,
            commentOnly: false,
            directives: false,
            labels: false,
            optOutput: true,
            binary: false,
            execute: false,
            demangle: false,
            libraryCode: false,
            trim: false,
            binaryObject: false,
            debugCalls: false,
        };
    }
}
