// Copyright (c) 2026, Compiler Explorer Authors
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

import {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import type {ToolResult} from '../../types/tool.interfaces.js';
import {parseRustOutput} from '../utils.js';
import {BaseTool} from './base-tool.js';

export class Co2MiriTool extends BaseTool {
    static get key() {
        return 'co2miri-tool';
    }

    override parseOutput(lines: string, inputFilename?: string, pathPrefix?: string): ResultLine[] {
        return parseRustOutput(lines, inputFilename, pathPrefix);
    }

    override async runTool(
        compilationInfo: CompilationInfo,
        inputFilepath?: string,
        args?: string[],
        stdin?: string,
        supportedLibraries?: any,
        dontAppendInputFilepath?: boolean,
    ): Promise<ToolResult> {
        const compilerId = compilationInfo.compiler.id;
        if (compilerId === 'co2cc') {
            return {
                id: this.tool.id,
                name: this.tool.name,
                code: -1,
                languageId: 'stderr',
                stderr: this.parseOutput('co2miri is only supported with co2rustc, not with co2cc (C compilation)'),
                stdout: [],
            };
        }
        return super.runTool(compilationInfo, inputFilepath, args, stdin, supportedLibraries, dontAppendInputFilepath);
    }
}
