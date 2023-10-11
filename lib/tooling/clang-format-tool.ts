// Copyright (c) 2019, Compiler Explorer Authors
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

import {ToolInfo} from '../../types/tool.interfaces.js';

import {BaseTool} from './base-tool.js';
import {ToolEnv} from './base-tool.interface.js';

export class ClangFormatTool extends BaseTool {
    static get key() {
        return 'clang-format-tool';
    }

    constructor(toolInfo: ToolInfo, env: ToolEnv) {
        super(toolInfo, env);

        this.addOptionsToToolArgs = false;
    }

    override async runTool(compilationInfo: Record<any, any>, inputFilepath: string, args: string[], stdin: string) {
        const sourcefile = inputFilepath;
        const compilerExe = compilationInfo.compiler.exe;
        const options = compilationInfo.options;
        const dir = path.dirname(sourcefile);

        let compileFlags = options.filter(option => option !== sourcefile);
        if (!compilerExe.includes('clang++')) {
            compileFlags = compileFlags.concat(this.tool.options);
        }

        await fs.writeFile(path.join(dir, 'compile_flags.txt'), compileFlags.join('\n'));
        const clang_format_file = path.join(dir, 'clang_format_options.txt');
        await fs.writeFile(clang_format_file, stdin);
        args.push(`-style=file:${clang_format_file}`);
        return await super.runTool(compilationInfo, sourcefile, args, stdin);
    }
}
