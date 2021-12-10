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

import * as utils from '../utils';

import { BaseTool } from './base-tool';

export class MicrosoftAnalysisTool extends BaseTool {
    static get key() { return 'microsoft-analysis-tool'; }

    constructor(toolInfo, env) {
        super(toolInfo, env);

        this.addOptionsToToolArgs = false;
    }

    async runTool(compilationInfo, inputFilepath, args, stdin, supportedLibraries) {
        const sourcefile = inputFilepath;
        const options = compilationInfo.options;
        const libOptions = super.getLibraryOptions(compilationInfo.libraries, supportedLibraries);
        const includeflags = super.getIncludeArguments(compilationInfo.libraries, supportedLibraries);
        // TODO: change these to /external:I instead of /I

        // order should be:
        //  1) options from the compiler config (compilationInfo.compiler.options)
        //  2) includes from the libraries (includeflags)
        //  3) options from the tool config (this.tool.options)
        //     -> before my patchup this was done only for non-clang compilers
        //  4) options manually specified in the compiler tab (options)
        //  5) options needed for analysis
        //  6) indepenent are `args` from the clang-tidy tab
        let compileFlags = utils.splitArguments(compilationInfo.compiler.options);
        compileFlags = compileFlags.concat(includeflags);
        compileFlags = compileFlags.concat(libOptions);

        const manualCompileFlags = options.filter(option => (option !== sourcefile));
        compileFlags = compileFlags.concat(manualCompileFlags);
        compileFlags = compileFlags.concat("/analyze:plugin EspXEngine.dll");
        // TODO: enable when using /external:I
        //compileFlags = compileFlags.concat("/external:W0 /analyze:external-");
        compileFlags = compileFlags.concat(this.tool.options);

        return await super.runTool(compilationInfo, sourcefile, args);
    }
}
