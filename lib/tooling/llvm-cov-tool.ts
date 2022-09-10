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

import fs from 'fs-extra';
import path from 'path';
import {logger} from '../logger';

import {BaseTool} from './base-tool';
import {ExecutionOptions} from '../../types/compilation/compilation.interfaces';

export class LLVMCovTool extends BaseTool {
    static get key() {
        return 'llvm-cov-tool';
    }

    override async runTool(compilationInfo, inputFilepath, args, stdin /*, supportedLibraries*/) {
        const compilationExecOptions = this.getDefaultExecOptions() as ExecutionOptions;
        compilationExecOptions.customCwd = path.dirname(inputFilepath);
        compilationExecOptions.input = stdin;
        try {
            const compilationResult = await this.exec(
                compilationInfo.compiler.exe,
                ['-fprofile-instr-generate', '-fcoverage-mapping', '-g', '-O0', inputFilepath, '-o', 'coverage.a'],
                compilationExecOptions,
            );

            if (compilationResult.code !== 0) {
                return this.createErrorResponse(
                    `<Compilation error>\n${compilationResult.stdout}\n${compilationResult.stderr}`,
                );
            }

            const runExecOptions = this.getDefaultExecOptions() as ExecutionOptions;
            runExecOptions.customCwd = path.dirname(inputFilepath);

            await this.exec('./coverage.a', [], {
                ...runExecOptions,
                input: stdin,
            });

            const folder = path.dirname(this.tool.exe);
            const profdataPath = path.join(folder, 'llvm-profdata');

            const profdataResult = await this.exec(
                profdataPath,
                ['merge', '-sparse', './default.profraw', '-o', './coverage.profdata'],
                runExecOptions,
            );
            if (profdataResult.code !== 0) {
                return this.createErrorResponse(
                    `<llvm-profdata error>\n${profdataResult.stdout}\n${profdataResult.stderr}`,
                );
            }

            const cppFiltPath = path.join(folder, 'llvm-cxxfilt');
            const covResult = await this.exec(
                this.tool.exe,
                [
                    'show',
                    './coverage.a',
                    '-instr-profile=./coverage.profdata',
                    '-format',
                    'text',
                    '-use-color',
                    '-Xdemangler',
                    cppFiltPath,
                    '-Xdemangler',
                    '-n',
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
            return this.createErrorResponse(`<Tool error: ${e}`);
        }
    }
}
