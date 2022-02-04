// Copyright (c) 2020, Compiler Explorer Authors
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

import { logger } from '../logger';
import * as ClangFormatTool from './clang-format-tool';
import { BaseTool } from './base-tool';

export class PreprocessorTool extends BaseTool {
    constructor(toolInfo, env) {
        super(toolInfo, env);

        this.reLine = /^#\s(\d*)\s(.*)\s?(\d*)/;

        this.clangformat = new ClangFormatTool(toolInfo, env);
    }

    async runCompiler(compilationInfo, inputFilepath, args) {
        let execOptions = this.getDefaultExecOptions();
        if (inputFilepath) execOptions.customCwd = path.dirname(inputFilepath);

        args = args ? args : [];
        if (inputFilepath) args.push(inputFilepath);

        const exeDir = path.dirname(compilationInfo.compiler.exe);

        try {
            const result = await this.exec(compilationInfo.compiler.exe, args, execOptions);
            return this.convertResult(result, inputFilepath, exeDir);
        } catch (e) {
            logger.error('Error while running tool: ', e);
            return this.createErrorResponse('Error while running tool');
        }
    }

    async enrich(result) {
        for (let line of result.stdout) {
            const match = line.text.match(this.reLine);
            if (match) {
                if (match[2].includes('example.cpp')) {
                    line.source = {
                        file: null,
                        line: parseInt(match[1]),
                    };
                }
            }
        }

        return result;
    }

    async runTool(compilationInfo, inputFilepath, args) {
        const sourceDir = path.dirname(inputFilepath);
        const outputFilename = 'preprocessor_output.txt';
        let preprocessArgs = ['-E', '-o', outputFilename];
        const filteredCompilerArgs = _.filter(compilationInfo.options, (arg) => {
            return (arg !== '-S');
        });
        const includeArgs = this.getIncludeArguments(compilationInfo.libraries, compilationInfo.compiler);
        preprocessArgs = preprocessArgs.concat(filteredCompilerArgs, includeArgs, args);

        const compilerResult = await this.runCompiler(compilationInfo, inputFilepath, preprocessArgs);
        if (compilerResult.code !== 0) {
            return compilerResult;
        } else {
            compilerResult.filenameTransform = () => inputFilepath;
            const formatResult = await this.clangformat.runTool(
                compilationInfo, path.join(sourceDir, outputFilename), [], '');
            return this.enrich(formatResult);
        }
    }
}
