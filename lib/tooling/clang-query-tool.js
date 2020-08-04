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
'use strict';

const
    fs = require('fs-extra'),
    path = require('path'),
    BaseTool = require('./base-tool');

class ClangQueryTool extends BaseTool {
    async runTool(compilationInfo, inputFilepath, args, stdin) {
        const sourcefile = inputFilepath;
        const compilerExe = compilationInfo.compiler.exe;
        const options = compilationInfo.options;
        const dir = path.dirname(sourcefile);

        const compileFlags = options.filter(option => (option !== sourcefile));
        if (!compilerExe.includes('clang++')) {
            compileFlags.push(this.tool.options);
        }

        await fs.writeFile(path.join(dir, 'compile_flags.txt'), compileFlags.join('\n'));
        await fs.writeFile(path.join(dir, 'query_commands.txt'), stdin);
        args.push('-f');
        args.push('query_commands.txt');
        const toolResult = await super.runTool(compilationInfo, sourcefile, args);

        if (toolResult.stdout.length > 0) {
            const lastLine = toolResult.stdout.length - 1;
            toolResult.stdout[lastLine].text = toolResult.stdout[lastLine].text.replace(/(clang-query>\s)/gi, '');
        }

        return toolResult;
    }
}

module.exports = ClangQueryTool;
