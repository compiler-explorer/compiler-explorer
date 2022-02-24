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

import { BaseCompiler } from '../base-compiler';
import { logger } from '../logger';

import { ErlangParser } from './argument-parsers';

export class ErlangCompiler extends BaseCompiler {
    static get key() { return 'erlang'; }

    /* eslint-disable no-unused-vars */
    optionsForFilter(filters, outputFilename) {
        return [
            '-noshell',
            '-eval',
            '{ok, Input} = init:get_argument(input),' +
                '{ok, _, Output} = compile:file(Input, [\'S\', binary, no_line_info, report]),' +
                `{ok,Fd} = file:open("${outputFilename}", [write]),` +
                'beam_listing:module(Fd, Output),' +
                'file:close(Fd),' +
                'halt().',
        ];
    }
    /* eslint-enable no-unused-vars */

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const result = await this.exec(compiler, options.concat(['-input', inputFilename]), execOptions);
        result.inputFilename = inputFilename;
        const transformedInput = result.filenameTransform(inputFilename);
        this.parseCompilationOutput(result, transformedInput);
        return result;
    }

    getVersion() {
        logger.info(`Gathering ${this.compiler.id} version information on ${this.compiler.exe}...`);
        if (this.compiler.explicitVersion) {
            logger.debug(`${this.compiler.id} has forced version output: ${this.compiler.explicitVersion}`);
            return {stdout: [this.compiler.explicitVersion], stderr: [], code: 0};
        }
        const execOptions = this.getDefaultExecOptions();
        const versionCmd = this.compilerProps(`compiler.${this.compiler.id}.runtime`);
        execOptions.timeoutMs = 0; // No timeout for --version. A sort of workaround for slow EFS/NFS on the prod site
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);

        try {
            return this.execCompilerCached(
                versionCmd,
                ['-noshell', '-eval', 'io:fwrite("~s~n", [erlang:system_info(otp_release)]), halt().'],
                execOptions,
            );
        } catch (err) {
            logger.error(`Unable to get version for compiler '${this.compiler.exe}' - ${err}`);
            return null;
        }
    }

    getOutputFilename(dirPath, outputFilebase) {
        return path.join(dirPath, `${outputFilebase}.S`);
    }

    getArgumentParser() {
        return ErlangParser;
    }
}
