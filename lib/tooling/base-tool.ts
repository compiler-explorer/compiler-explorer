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

import PromClient from 'prom-client';
import _ from 'underscore';

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {UnprocessedExecResult} from '../../types/execution/execution.interfaces';
import {Library, SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces';
import {ResultLine} from '../../types/resultline/resultline.interfaces';
import * as exec from '../exec';
import {logger} from '../logger';
import {parseOutput} from '../utils';

import {Tool, ToolEnv, ToolInfo, ToolResult, ToolTypeKey} from './base-tool.interface';

const toolCounter = new PromClient.Counter({
    name: 'tool_invocations_total',
    help: 'Number of tool invocations',
    labelNames: ['language', 'name'],
});

export class BaseTool implements Tool {
    public readonly tool: ToolInfo;
    private env: ToolEnv;
    protected addOptionsToToolArgs = true;

    constructor(toolInfo: ToolInfo, env: ToolEnv) {
        this.tool = toolInfo;
        this.env = env;
        this.addOptionsToToolArgs = true;
    }

    getId() {
        return this.tool.id;
    }

    getType(): ToolTypeKey {
        return this.tool.type || 'independent';
    }

    getUniqueFilePrefix() {
        const timestamp = process.hrtime();
        const timestamp_str = '_' + timestamp[0] * 1000000 + timestamp[1] / 1000;
        return this.tool.id.replace(/[^\da-z]/gi, '_') + timestamp_str + '_';
    }

    isCompilerExcluded(compilerId: string, compilerProps: ToolEnv['compilerProps']): boolean {
        if (this.tool.includeKey) {
            // If the includeKey is set, we only support compilers that have a truthy 'includeKey'.
            if (!compilerProps(this.tool.includeKey)) {
                return true;
            }
            // Even if the include key is truthy, we fall back to the exclusion list.
        }
        return this.tool.exclude.find(excl => compilerId.includes(excl)) !== undefined;
    }

    exec(toolExe: string, args: string[], options: ExecutionOptions) {
        return exec.execute(toolExe, args, options);
    }

    getDefaultExecOptions(): ExecutionOptions {
        return {
            timeoutMs: this.env.ceProps('compileTimeoutMs', 7500) as number,
            maxErrorOutput: this.env.ceProps('max-error-output', 5000) as number,
            wrapper: this.env.compilerProps('compiler-wrapper'),
        };
    }

    // By default calls utils.parseOutput, but lets subclasses override their output processing
    protected parseOutput(lines: string, inputFilename?: string, pathPrefix?: string): ResultLine[] {
        return parseOutput(lines, inputFilename, pathPrefix);
    }

    createErrorResponse(message: string): ToolResult {
        return {
            id: this.tool.id,
            name: this.tool.name,
            code: -1,
            languageId: 'stderr',
            stdout: [],
            stderr: this.parseOutput(message),
        };
    }

    // mostly copy&paste from base-compiler.js
    findLibVersion(selectedLib: SelectedLibraryVersion, supportedLibraries: Record<string, Library>) {
        const foundLib = _.find(supportedLibraries, (o, libId) => libId === selectedLib.id);
        if (!foundLib) return false;

        return _.find(foundLib.versions, (o, versionId) => versionId === selectedLib.version);
    }

    // mostly copy&paste from base-compiler.js
    getIncludeArguments(libraries: SelectedLibraryVersion[], supportedLibraries: Record<string, Library>): string[] {
        const includeFlag = '-I';

        return libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib, supportedLibraries);
            if (!foundVersion) return [];

            return foundVersion.path.map(path => includeFlag + path);
        });
    }

    getLibraryOptions(libraries: SelectedLibraryVersion[], supportedLibraries: Record<string, Library>): string[] {
        return libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib, supportedLibraries);
            if (!foundVersion) return [];

            return foundVersion.options;
        });
    }

    async runTool(
        compilationInfo: Record<any, any>,
        inputFilepath?: string,
        args?: string[],
        stdin?: string,
        supportedLibraries?: Record<string, Library>,
    ) {
        if (this.tool.name) {
            toolCounter.inc({
                language: compilationInfo.compiler.lang,
                name: this.tool.name,
            });
        }
        const execOptions = this.getDefaultExecOptions();
        if (inputFilepath) execOptions.customCwd = path.dirname(inputFilepath);
        execOptions.input = stdin;

        args = args ? args : [];
        if (this.addOptionsToToolArgs) args = this.tool.options.concat(args);
        if (inputFilepath) args.push(inputFilepath);

        const exeDir = path.dirname(this.tool.exe);

        try {
            const result = await this.exec(this.tool.exe, args, execOptions);
            return this.convertResult(result, inputFilepath, exeDir);
        } catch (e) {
            logger.error('Error while running tool: ', e);
            return this.createErrorResponse('Error while running tool');
        }
    }

    convertResult(result: UnprocessedExecResult, inputFilepath?: string, exeDir?: string): ToolResult {
        const transformedFilepath = inputFilepath ? result.filenameTransform(inputFilepath) : undefined;
        return {
            id: this.tool.id,
            name: this.tool.name,
            code: result.code,
            languageId: this.tool.languageId,
            stderr: this.parseOutput(result.stderr, transformedFilepath, exeDir),
            stdout: this.parseOutput(result.stdout, transformedFilepath, exeDir),
        };
    }
}
