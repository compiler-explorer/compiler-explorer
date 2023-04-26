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

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';

import {BaseTool} from './base-tool.js';

export class LLVMCovTool extends BaseTool {
    static get key() {
        return 'llvm-cov-tool';
    }

    override async runTool(compilationInfo, inputFilepath, args: string[], stdin) {
        const compilationExecOptions = this.getDefaultExecOptions();
        compilationExecOptions.customCwd = path.dirname(inputFilepath);
        compilationExecOptions.input = stdin;
        try {
            const generatedExecutableName = this.getUniqueFilePrefix() + '-coverage.a';
            // Remove inputs
            const options = compilationInfo.compilationOptions.slice().splice(0, 4);
            const compilationArgs = [
                ...options,
                '-fprofile-instr-generate',
                '-fcoverage-mapping',
                '-g',
                '-O0',
                inputFilepath,
                '-o',
                generatedExecutableName,
            ];

            const compilationResult = await this.exec(
                compilationInfo.compiler.exe,
                compilationArgs,
                compilationExecOptions,
            );

            if (compilationResult.code !== 0) {
                return this.createErrorResponse(
                    `<Compilation error>\n${compilationResult.stdout}\n${compilationResult.stderr}`,
                );
            }

            const runExecOptions = this.getDefaultExecOptions() as ExecutionOptions;
            runExecOptions.customCwd = path.dirname(inputFilepath);

            await this.exec('./' + generatedExecutableName, [], {
                ...runExecOptions,
                input: stdin,
            });

            const folder = path.dirname(this.tool.exe);
            const profdataPath = path.join(folder, 'llvm-profdata');

            const generatedProfdataName = this.getUniqueFilePrefix() + '.profdata';
            const profdataResult = await this.exec(
                profdataPath,
                ['merge', '-sparse', './default.profraw', '-o', './' + generatedProfdataName],
                runExecOptions,
            );
            if (profdataResult.code !== 0) {
                return this.createErrorResponse(
                    `<llvm-profdata error>\n${profdataResult.stdout}\n${profdataResult.stderr}`,
                );
            }

            const covResult = await this.exec(
                this.tool.exe,
                [
                    'show',
                    './' + generatedExecutableName,
                    '-instr-profile=./' + generatedProfdataName,
                    '-format',
                    'text',
                    '-use-color',
                    '-compilation-dir=./',
                    ...args,
                ],
                runExecOptions,
            );
            if (covResult.code === 0) {
                return this.convertResult(covResult, inputFilepath, path.dirname(this.tool.exe));
            } else {
                return this.createErrorResponse(`<llvm-cov error>\n${covResult.stdout}\n${covResult.stderr}`);
            }
        } catch (e) {
            return this.createErrorResponse(`<Tool error: ${e}>`);
        }
    }
}
