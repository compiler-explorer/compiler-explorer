// Copyright (c) 2021, Compiler Explorer Authors
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

import {ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';

import {CircleParser} from './argument-parsers.js';

export class CircleCompiler extends BaseCompiler {
    static get key() {
        return 'circle';
    }

    protected override getArgumentParserClass() {
        return CircleParser;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = [`-o=${this.filename(outputFilename)}`];
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }
        if (!filters.binary) {
            options.push('-S');
        }
        return options;
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        // Do not add '.s' as default implementation does,
        // because circle emit assembly file instead of executable.
        return path.join(dirPath, outputFilebase);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);

        return await super.runCompiler(compiler, options, inputFilename, execOptions);
    }
}
