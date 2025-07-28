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

import fs from 'node:fs/promises';
import path from 'node:path';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {changeExtension} from '../utils.js';

export class MojoCompiler extends BaseCompiler {
    static get key() {
        return 'mojo';
    }

    constructor(info, env) {
        super(info, env);
        this.delayCleanupTemp = true;
        this.compiler.supportsOptOutput = true;
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit', 'llvm'];
        this.compiler.minIrArgs = ['--emit=llvm'];
    }

    override getOutputFilename(dirPath: string, inputFileBase: string) {
        // This method tells CE where to find the assembly output
        const outputPath = path.join(dirPath, inputFileBase + '.s');
        return outputPath;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions: string[]) {
        if (filters.binary) return ['build'];
        return ['build', '--emit=asm', '-o', outputFilename];
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string) {
        const executablePath = path.join(dirPath, 'example');
        return executablePath;
    }

    override getIrOutputFilename(inputFilename: string): string {
        return inputFilename.replace(/\.mojo$/, '.ll');
    }

    override async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: any,
        produceCfg: boolean,
        filters: any,
    ) {
        // Remove -o/--emit and their values from options
        const filteredOptions: string[] = options.filter(
            (opt, idx, arr) =>
                opt !== '-o' &&
                opt !== '--emit' &&
                !opt.startsWith('--emit=') &&
                arr[idx - 1] !== '-o' &&
                arr[idx - 1] !== '--emit',
        );
        // Compute the .ll output path
        const llPath = changeExtension(inputFilename, '.ll');
        const newOptions = [...filteredOptions, '--emit=llvm', '-o', llPath];
        const execOptions = this.getDefaultExecOptions();
        execOptions.maxOutput = 1024 * 1024 * 1024;
        const output = await this.runCompiler(this.compiler.exe, newOptions, this.compileFilename, execOptions);
        if (output.code !== 0) {
            console.error('Mojo IR build failed:', newOptions, output.stderr, output.stdout);
            return {asm: [{text: 'Failed to run compiler to get IR code'}]};
        }
        const irText = await fs.readFile(llPath, 'utf8');
        return {asm: irText.split('\n').map(text => ({text}))};
    }
}
