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

import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {ParseFilters} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';

export class RacketCompiler extends BaseCompiler {
    private raco: string;

    static get key() {
        return 'racket';
    }

    constructor(info, env) {
        // Disable output filters, as they currently don't do anything
        if (!info.disabledFilters) {
            info.disabledFilters = ['labels', 'directives', 'commentOnly', 'trim'];
        }
        super(info, env);
        this.raco = this.compilerProps(`compiler.${this.compiler.id}.raco`);
    }

    override optionsForFilter(filters: ParseFilters, outputFilename: any, userOptions?: any): string[] {
        return [];
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, `${outputFilebase}.decompiled.rkt`);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        // Compile via `raco make`
        options.unshift('make');
        const makeResult = await this.exec(this.raco, options, execOptions);
        // TODO: Check `raco make` result

        // Retrieve assembly via `raco decompile` with `disassemble` package
        const outputFilename = path.join(inputFilename, '..', `${this.outputFilebase}.decompiled.rkt`);
        const decompilePipeline = `${this.raco} decompile '${inputFilename}' > '${outputFilename}'`;
        const decompileResult = await this.exec('bash', ['-c', decompilePipeline], execOptions);

        return this.transformToCompilationResult(decompileResult, inputFilename);
    }
}
