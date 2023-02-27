// Copyright (c) 2018, Compiler Explorer Authors
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

import {ClangParser} from './argument-parsers.js';

export class DMDCompiler extends BaseCompiler {
    static get key() {
        return 'dmd';
    }

    constructor(compiler: PreliminaryCompilerInfo, env) {
        super(compiler, env);
        this.compiler.supportsIntel = true;
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options = ['-g', '-of' + this.filename(outputFilename)];
        if (filters.binaryObject) options.push('-c');
        return options;
    }

    override async execPostProcess(result, postProcesses, outputFilename, maxSize) {
        const dirPath = path.dirname(outputFilename);
        const lPath = path.basename(outputFilename);
        return this.handlePostProcessResult(
            result,
            await this.exec(postProcesses[0], ['-l', lPath], {customCwd: dirPath, maxOutput: maxSize}),
        );
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, `${outputFilebase}.s`);
    }

    override buildExecutable(compiler, options, inputFilename, execOptions) {
        options = options.filter(param => param !== '-c');

        return this.runCompiler(compiler, options, inputFilename, execOptions);
    }

    override getArgumentParser() {
        return ClangParser;
    }

    override filterUserOptions(userOptions: string[]) {
        return userOptions.filter(option => option !== '-run' && option !== '-man' && !option.startsWith('-Xf'));
    }
}
