// Copyright (c) 2025, Compiler Explorer Authors
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

import {CompilationInfo, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import {ToolResult} from '../../types/tool.interfaces.js';
import {OptionsHandlerLibrary} from '../options-handler.js';
import {BaseTool} from './base-tool.js';

export class ExplainTool extends BaseTool {
    static get key() {
        return 'explain-tool';
    }

    override async runTool(
        compilationInfo: CompilationInfo,
        inputFilepath?: string,
        args?: string[],
        stdin?: string,
        supportedLibraries?: Record<string, OptionsHandlerLibrary>,
        dontAppendInputFilepath?: boolean,
    ): Promise<ToolResult> {
        // Get the API endpoint from configuration
        const apiEndpoint = this.env.ceProps('explainApiEndpoint');

        if (!apiEndpoint) {
            throw new Error('Claude Explain API endpoint not configured (explainApiEndpoint)');
        }

        // Return a minimal result with the API endpoint as custom data
        return {
            id: this.tool.id,
            name: this.tool.name,
            code: 0,
            stdout: [],
            stderr: [],
            // Add API endpoint as a custom field (will be preserved in JSON)
            explainApiEndpoint: apiEndpoint,
        } as ToolResult & {explainApiEndpoint: string};
    }

    // We don't need to execute anything for this tool
    // as it's just a placeholder for the client-side implementation
    override exec(toolExe: string, args: string[], options: ExecutionOptions): Promise<UnprocessedExecResult> {
        return Promise.resolve({
            code: 0,
            stdout: '',
            stderr: '',
            timedOut: false,
            okToCache: true,
            filenameTransform: (x: string) => x,
            execTime: 0,
            truncated: false,
        });
    }
}
