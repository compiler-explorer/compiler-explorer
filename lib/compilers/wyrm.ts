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
import fs from 'fs/promises';
import path from 'path';

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class WyrmCompiler extends BaseCompiler {
    static get key() {
        return 'wyrm';
    }

    gcc: BaseCompiler | undefined;
    gccId: string;

    constructor(compilerInfo: PreliminaryCompilerInfo & {disabledFilters?: string[]}, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.gccId = this.compilerProps<string>(`group.${this.compiler.group}.gccId`);
    }

    getGcc(): BaseCompiler {
        if (!this.gcc) {
            this.gcc = unwrap(global.handler_config.compileHandler.findCompiler('c', this.gccId));
        }
        return unwrap(this.gcc);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
        filters?: ParseFiltersAndOutputOptions,
    ) {
        const gcc = this.getGcc();
        const result = await gcc.runCompiler(
            gcc.getInfo().exe,
            options.concat(`-fplugin=${compiler}`, '-fgimple'),
            inputFilename,
            execOptions,
        );
        const oPath = options[options.indexOf('-o') + 1];
        await fs.rename(`${path.dirname(oPath)}/x.ll`, oPath);
        return {
            ...result,
            languageId: this.getCompilerResultLanguageId(filters),
        };
    }

    override async getVersion() {
        return {
            stdout: ['trunk'],
            stderr: [],
        };
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return 'llvm-ir';
    }
}
