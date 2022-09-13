// Copyright (c) 2018, Mitch Kennedy
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

import fs from 'fs-extra';

import {BaseCompiler} from '../base-compiler';
import * as utils from '../utils';

export class AdaCompiler extends BaseCompiler {
    static get key() {
        return 'ada';
    }

    constructor(info, env) {
        super(info, env);
        this.compiler.supportsGccDump = true;
        this.compiler.removeEmptyGccDump = true;

        // used for all GNAT related panes (Expanded code, Tree)
        this.compiler.supportsGnatDebugViews = true;
    }

    getExecutableFilename(dirPath) {
        // The name here must match the value used in the pragma Source_File
        // in the user provided source.
        return path.join(dirPath, 'example');
    }

    getOutputFilename(dirPath) {
        // The basename here must match the value used in the pragma Source_File
        // in the user provided source.
        return path.join(dirPath, 'example.out');
    }

    prepareArguments(userOptions, filters, backendOptions, inputFilename, outputFilename, libraries) {
        backendOptions = backendOptions || {};

        // super call is needed as it handles the GCC Dump files.
        let backend_opts = super.optionsForBackend(backendOptions, outputFilename);

        // gnatmake  opts  name  {[-cargs opts] [-bargs opts] [-largs opts] [-margs opts]}
        //                 ^                 ^    ^             ^             ^
        //                 |                 |    |             |             |
        //                 `- inputFilename  |    |             |             `-- for gnatmake
        //                                   |    |             `-- for linker
        //                                   |    `-- for binder (unused here)
        //                                   `-- for compiler (gcc)

        let gnatmake_opts = [];
        let compiler_opts = [];
        let binder_opts = [];
        let linker_opts = [''];

        if (this.compiler.adarts) {
            gnatmake_opts.push(`--RTS=${this.compiler.adarts}`);
        }

        if (backendOptions.produceGnatDebug && this.compiler.supportsGnatDebugViews)
            // This is using stdout
            gnatmake_opts.push('-gnatGL');

        if (backendOptions.produceGnatDebugTree && this.compiler.supportsGnatDebugViews)
            // This is also using stdout
            gnatmake_opts.push('-gnatdt');

        gnatmake_opts.push(
            '-g',
            '-fdiagnostics-color=always',
            '-eS', // output commands to stdout, they are not errors
            inputFilename,
        );

        if (!filters.binary) {
            gnatmake_opts.push(
                '-S', // Generate ASM
                '-c', // Compile only
                '-fverbose-asm', // Generate verbose ASM showing variables
            );

            // produce assembly output in outputFilename
            compiler_opts.push('-o', outputFilename);

            if (this.compiler.intelAsm && filters.intel) {
                for (const opt of this.compiler.intelAsm.split(' ')) {
                    gnatmake_opts.push(opt);
                }
            }
        } else {
            gnatmake_opts.push('-o', outputFilename);
        }

        // Spread the options coming from outside (user, backend or config options)
        // in the correct list.

        let part = 0; // 0: gnatmake, 1: compiler, 2: linker, 3: binder
        for (const a of backend_opts.concat(utils.splitArguments(this.compiler.options), userOptions)) {
            if (a === '-cargs') {
                part = 1;
                continue;
            } else if (a === '-largs') {
                part = 2;
                continue;
            } else if (a === '-bargs') {
                part = 3;
                continue;
            } else if (a === '-margs') {
                part = 0;
                continue;
            }

            if (part === 0) {
                gnatmake_opts.push(a);
            } else if (part === 1) {
                compiler_opts.push(a);
            } else if (part === 2) {
                linker_opts.push(a);
            } else if (part === 3) {
                binder_opts.push(a);
            }
        }

        return gnatmake_opts.concat('-cargs', compiler_opts, '-largs', linker_opts, '-bargs', binder_opts);
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const appHome = path.dirname(inputFilename);

        // create a subdir so that files automatically created by GNAT don't
        // conflict with anything else in parent dir.
        const temp_dir = path.join(appHome, 'tempsub');
        await fs.mkdir(temp_dir);

        // Set the working directory to be the temp directory that has been created
        execOptions.appHome = appHome;
        execOptions.customCwd = temp_dir;

        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;

        const baseFilename = path.basename(inputFilename);
        result.stdout = utils.parseOutput(result.stdout, baseFilename, execOptions.customCwd);
        result.stderr = utils.parseOutput(result.stderr, baseFilename, execOptions.customCwd);
        return result;
    }
}
