// Copyright (c) 2018, Mitch Kennedy
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

import { BaseCompiler } from '../base-compiler';
import * as utils from '../utils';

export class AdaCompiler extends BaseCompiler {
    static get key() { return 'ada'; }

    constructor(info, env) {
        super(info, env);
        this.compiler.supportsGccDump = true;
        this.compiler.removeEmptyGccDump = true;
        this.compiler.supportsIntel = true;
        this.compiler.supportsGnatDebugView = true;
    }

    getExecutableFilename(dirPath) {
        return path.join(dirPath, 'example');
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, 'example.o');
    }

    optionsForBackend(backendOptions, outputFilename){
        // super is needed as it handles the GCC Dump files.
        const opts = super.optionsForBackend (backendOptions, outputFilename);

        if (backendOptions.produceGnatDebug && this.compiler.supportsGnatDebugView)
            opts.push('-gnatDGL');

        return opts;
    }

    optionsForFilter(filters, outputFilename) {
        let options = [];

        if (!filters.binary) {
            options.push(
                'compile',
                '-g', // enable debugging
                '-S', // Generate ASM
                '-fdiagnostics-color=always',
                '-fverbose-asm', // Generate verbose ASM showing variables
                '-c', // Compile only
                '-eS', // commands are not errors
                '-cargs', // Compiler Switches for gcc.
                '-o',
                outputFilename,
            );

            if (this.compiler.intelAsm && filters.intel) {
                options = options.concat(this.compiler.intelAsm.split(' '));
            }
        } else {
            options.push(
                'make',
                '-eS',
                '-g',
                'example',
                '-cargs',
            );
        }

        return options;
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        // Set the working directory to be the temp directory that has been created
        execOptions.customCwd = path.dirname(inputFilename);

        // As the file name is always appended to the end of the options array we need to
        // find where the '-cargs' flag is in options. This is to allow us to set the
        // output as 'output.s' and not end up with 'example.s'. If the output is left
        // as 'example.s' CE can't find it and thus you get no output.
        const inputFileName = options.pop();
        for (let i = 0; i < options.length; i++) {
            if (options[i] === '-cargs') {
                options.splice(i, 0, inputFileName);
                break;
            }
        }

        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;

        const baseFilename = path.basename(inputFileName);
        result.stdout = utils.parseOutput(result.stdout, baseFilename, execOptions.customCwd);
        result.stderr = utils.parseOutput(result.stderr, baseFilename, execOptions.customCwd);
        return result;
    }
}
