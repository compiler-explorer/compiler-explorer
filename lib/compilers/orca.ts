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

import path from 'node:path';

import {BypassCache, CacheKey, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {ORCAAsmParser} from '../parsers/asm-parser-orca.js';
import * as utils from '../utils.js';

export class ORCACompiler extends BaseCompiler {
    static get key() {
        return 'orca';
    }

    goldenGate: string;

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.goldenGate = this.compilerProps<string>(`compiler.${this.compiler.id}.goldenGate`);
        this.externalparser = null;
        this.asm = new ORCAAsmParser(this.compilerProps);
        this.compiler.demangler = '';
        this.demanglerClass = null;
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.goldenGate) {
            execOptions.env.GOLDEN_GATE = this.goldenGate;
        }

        return execOptions;
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, `${outputFilebase}.a`);
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        let options: string[];

        if (filters.binary) {
            options = ['cmpl', 'keep=' + this.filename(outputFilename)];
        } else {
            filters.binaryObject = true;
            options = ['compile', 'keep=' + this.filename(utils.changeExtension(outputFilename, ''))];
        }

        return options;
    }

    override async handleExecution(
        key: CacheKey,
        executeParameters: ExecutableExecutionOptions,
        bypassCache: BypassCache,
    ): Promise<CompilationResult> {
        if (this.goldenGate) {
            executeParameters.env.GOLDEN_GATE = this.goldenGate;
        }
        return super.handleExecution(key, executeParameters, bypassCache);
    }
}
