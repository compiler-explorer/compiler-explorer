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

import path from 'node:path';

import type {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import type {OptionsHandlerLibrary} from '../options-handler.js';
import * as temp from '../temp.js';
import {ce_temp_prefix, parseRustOutput} from '../utils.js';

import {BaseTool} from './base-tool.js';

export class ClippyTool extends BaseTool {
    static get key() {
        return 'clippy-tool';
    }

    override getToolExe(compilationInfo: CompilationInfo): string {
        return path.format({dir: path.dirname(compilationInfo.compiler.exe), base: 'clippy-driver'});
    }

    override async getCustomCwd(inputFilepath: string): Promise<string> {
        return await temp.mkdir(ce_temp_prefix);
    }

    override async runTool(
        compilationInfo: CompilationInfo,
        inputFilepath?: string,
        args?: string[],
        stdin?: string,
        supportedLibraries?: Record<string, OptionsHandlerLibrary>,
    ) {
        const clippyArgs = ['--color=always', ...(args || [])];
        return await super.runTool(compilationInfo, inputFilepath, clippyArgs, stdin, supportedLibraries);
    }

    override parseOutput(lines: string, inputFilename?: string, pathPrefix?: string): ResultLine[] {
        return parseRustOutput(lines, inputFilename, pathPrefix);
    }
}
