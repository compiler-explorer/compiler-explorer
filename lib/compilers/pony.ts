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

import _ from 'underscore';

import type {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {unwrap} from '../assert.js';

export class PonyCompiler extends BaseCompiler {
    static get key() {
        return 'pony';
    }

    /* constructor(info: any, env: any) {
        super(info, env);

        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--pass', 'ir'];
    } */

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: any, userOptions?: any): string[] {
        let options = ['-d', '-b', path.parse(outputFilename).name];

        if (!filters.binary) {
            options = options.concat(['--pass', 'asm']);
        }

        return options;
    }

    override preProcess(source: string, filters: any) {
        // I do not think you can make a Pony "library", so you must always have a main.
        // Looking at the stdlib, the main is used as a test harness.
        if (!this.stubRe.test(source)) {
            source += `\n${this.stubText}\n`;
        }
        return source;
    }

    override async generateIR(inputFilename: string, options: string[], filters: ParseFiltersAndOutputOptions) {
        const newOptions = _.filter(options, option => !['--pass', 'asm'].includes(option)).concat(
            unwrap(this.compiler.irArg),
        );

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const output = await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions);
        if (output.code !== 0) {
            return [{text: 'Failed to run compiler to get IR code'}];
        }
        const ir = await this.processIrOutput(output, filters);
        return ir.asm;
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

        // Pony operates upon the directory as a whole, not files it seems
        // So we must set the input to the directory rather than a file.
        options = _.map(options, arg => (arg.includes(inputFilename) ? path.dirname(arg) : arg));

        const compilerExecResult = await this.exec(compiler, options, execOptions);
        return this.transformToCompilationResult(compilerExecResult, inputFilename);
    }
}
