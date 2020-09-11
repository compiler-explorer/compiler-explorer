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
'use strict';

const
    _ = require('underscore'),
    path = require('path'),
    logger = require('../logger'),
    ClangFormatTool = require('./clang-format-tool'),
    BaseTool = require('./base-tool');

class PreprocessorTool extends BaseTool {
    constructor(toolInfo, env) {
        super(toolInfo, env);

        this.clangformat = new ClangFormatTool(toolInfo, env);
    }

    async runCompiler(compilationInfo, inputFilepath, args, stdin) {
        let execOptions = this.getDefaultExecOptions();
        if (inputFilepath) execOptions.customCwd = path.dirname(inputFilepath);
        execOptions.input = stdin;

        args = args ? args : [];
        if (inputFilepath) args.push(inputFilepath);

        const exeDir = path.dirname(compilationInfo.exe);

        try {
            const result = await this.exec(compilationInfo.exe, args, execOptions);
            return this.convertResult(result, inputFilepath, exeDir);
        } catch (e) {
            logger.error('Error while running tool: ', e);
            return this.createErrorResponse('Error while running tool');
        }
    }

    async runTool(compilationInfo, inputFilepath, args) {
        const preprocessArgs = ['-E'];
        const filteredCompilerArgs = _.filter(compilationInfo.options, (arg) => {
            return (arg !== '-S');
        });
        preprocessArgs.concat(filteredCompilerArgs);
        preprocessArgs.concat(args);

        //const compilerResult = await runCompiler(compilationInfo, inputFilepath, args, stdin);

        //this.clangformat.runTool(compilationInfo, )
    }
}

module.exports = PreprocessorTool;
