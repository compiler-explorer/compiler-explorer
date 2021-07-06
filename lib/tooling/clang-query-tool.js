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

import fs from 'fs-extra';

import { BaseTool } from './base-tool';

export class ClangQueryTool extends BaseTool {
    static get key() { return 'clang-query-tool'; }

    constructor(toolInfo, env) {
        super(toolInfo, env);

        this.addOptionsToToolArgs = false;
    }

    async runTool(compilationInfo, inputFilepath, args, stdin) {
        const sourcefile = inputFilepath;
        const compilerExe = compilationInfo.compiler.exe;
        const options = compilationInfo.options;
        const dir = path.dirname(sourcefile);

        const compileFlags = options.filter(option => (option !== sourcefile));
        if (!compilerExe.includes('clang++')) {
            compileFlags.push(this.tool.options);
        }

        const query_commands_file = this.getUniqueFilePrefix() + 'query_commands.txt';

        await fs.writeFile(path.join(dir, 'compile_flags.txt'), compileFlags.join('\n'));
        await fs.writeFile(path.join(dir, query_commands_file), stdin);
        args.push('-f', query_commands_file);
        const toolResult = await super.runTool(compilationInfo, sourcefile, args);

        if (toolResult.stdout.length > 0) {
            const lastLine = toolResult.stdout.length - 1;
            toolResult.stdout[lastLine].text = toolResult.stdout[lastLine].text.replace(/(clang-query>\s)/gi, '');
        }

        return toolResult;
    }
}
