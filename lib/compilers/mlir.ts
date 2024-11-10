// Copyright (c) 2022, Compiler Explorer Authors
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

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

import {BaseParser} from './argument-parsers.js';

function isMlirTranslate(compilerInfo: PreliminaryCompilerInfo): boolean {
    return compilerInfo.group === 'mlirtranslate';
}

export class MLIRCompiler extends BaseCompiler {
    static get key() {
        return 'mlir';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(
            {
                disabledFilters: isMlirTranslate(compilerInfo)
                    ? []
                    : [
                          'binary',
                          'execute',
                          'demangle',
                          'intel',
                          'labels',
                          'libraryCode',
                          'directives',
                          'commentOnly',
                          'trim',
                          'debugCalls',
                      ],
                ...compilerInfo,
            },
            env,
        );
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'mlir';
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, 'example.out.mlir');
    }

    override optionsForBackend(backendOptions: Record<string, any>, outputFilename: string): string[] {
        return ['-o', outputFilename];
    }

    override getArgumentParserClass(): any {
        return BaseParser;
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): any[] {
        return [];
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions, options: string[]) {
        // at some point maybe a custom parser can be written, for now just don't filter anything
        return super.processAsm(
            result,
            isMlirTranslate(this.compiler)
                ? filters
                : {
                      labels: false,
                      binary: false,
                      commentOnly: false,
                      demangle: false,
                      optOutput: false,
                      directives: false,
                      dontMaskFilenames: false,
                      execute: false,
                      intel: false,
                      libraryCode: false,
                      trim: false,
                      debugCalls: false,
                  },
            options,
        );
    }
}
