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

import {splitArguments} from '../../shared/common-utils.js';
import {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import {ToolInfo} from '../../types/tool.interfaces.js';
import {OptionsHandlerLibrary} from '../options-handler.js';

import {ToolEnv} from './base-tool.interface.js';
import {BaseTool} from './base-tool.js';

export class BrontoRefactorTool extends BaseTool {
    static get key() {
        return 'bronto-refactor-tool';
    }

    constructor(toolInfo: ToolInfo, env: ToolEnv) {
        super(toolInfo, env);

        this.addOptionsToToolArgs = false;
    }

    override async runTool(
        compilationInfo: CompilationInfo,
        inputFilepath: string,
        args: string[],
        _stdin?: string,
        supportedLibraries?: Record<string, OptionsHandlerLibrary>,
    ) {
        const sourcefile = inputFilepath;
        const options = compilationInfo.options;
        const includeflags = super.getIncludeArguments(compilationInfo.libraries, supportedLibraries || {});
        const libOptions = super.getLibraryOptions(compilationInfo.libraries, supportedLibraries || {});

        let compileFlags = ['compiler-explorer']
            .concat(args)
            .concat([sourcefile, '--clang-path', compilationInfo.compiler.exe, '--'])
            .concat(splitArguments(compilationInfo.compiler.options));
        compileFlags = compileFlags.concat(includeflags);
        compileFlags = compileFlags.concat(libOptions);

        const manualCompileFlags = options.filter(option => option !== sourcefile);
        compileFlags = compileFlags.concat(manualCompileFlags);
        return super.runTool(compilationInfo, sourcefile, compileFlags);
    }
}
