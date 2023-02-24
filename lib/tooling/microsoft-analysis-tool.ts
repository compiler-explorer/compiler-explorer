// Copyright (c) 2023, Compiler Explorer Authors
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

import {Library} from '../../types/libraries/libraries.interfaces';
import {ToolInfo} from '../../types/tool.interfaces';
import {unwrap} from '../assert';
import {logger} from '../logger';
import * as utils from '../utils';

import {BaseTool} from './base-tool';
import {ToolEnv} from './base-tool.interface';

export class MicrosoftAnalysisTool extends BaseTool {
    static get key() {
        return 'microsoft-analysis-tool';
    }

    constructor(toolInfo: ToolInfo, env: ToolEnv) {
        super(toolInfo, env);

        this.addOptionsToToolArgs = false;
    }

    async runCompilerTool(compilationInfo: Record<any, any>, inputFilepath?: string, args?: string[], stdin?: string) {
        const execOptions = this.getDefaultExecOptions();
        if (inputFilepath) execOptions.customCwd = path.dirname(inputFilepath);
        execOptions.input = stdin;

        args = args || [];
        if (this.addOptionsToToolArgs) args = this.tool.options.concat(args);
        if (inputFilepath) args.push(inputFilepath);

        const exeDir = path.dirname(compilationInfo.compiler.exe);

        execOptions.env = Object.assign({}, execOptions.env);

        if (compilationInfo.compiler.includePath) {
            execOptions.env['INCLUDE'] = compilationInfo.compiler.includePath;
        }
        if (compilationInfo.compiler.libPath) {
            execOptions.env['LIB'] = compilationInfo.compiler.libPath.join(';');
        }
        for (const [env, to] of compilationInfo.compiler.envVars) {
            execOptions.env[env] = to;
        }

        try {
            const result = await this.exec(compilationInfo.compiler.exe, args, execOptions);
            return this.convertResult(result, inputFilepath, exeDir);
        } catch (e) {
            logger.error('Error while running tool: ', e);
            return this.createErrorResponse('Error while running tool');
        }
    }

    override async runTool(
        compilationInfo: Record<any, any>,
        inputFilepath?: string,
        args?: string[],
        stdin?: string,
        supportedLibraries?: Record<string, Library>
    ) {
        const sourcefile = inputFilepath;
        const options = compilationInfo.options;
        const libOptions = super.getLibraryOptions(compilationInfo.libraries, unwrap(supportedLibraries));
        const includeflags = super.getIncludeArguments(compilationInfo.libraries, unwrap(supportedLibraries));

        let compileFlags = utils.splitArguments(compilationInfo.compiler.options);
        compileFlags = compileFlags.concat(includeflags, libOptions);

        const manualCompileFlags = options.filter(option => option !== sourcefile);
        compileFlags = compileFlags.concat(
            manualCompileFlags,
            '/nologo',
            '/analyze:plugin',
            'EspXEngine.dll',
            '/analyze:external-',
            '/external:env:INCLUDE',
            '/external:W0',
            this.tool.options
        );

        return await this.runCompilerTool(compilationInfo, sourcefile, compileFlags, stdin);
    }
}
