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

import fs from 'node:fs/promises';
import path from 'node:path';

import {splitArguments} from '../../shared/common-utils.js';
import {CompilationInfo} from '../../types/compilation/compilation.interfaces.js';
import {ToolResult} from '../../types/tool.interfaces.js';
import {logger} from '../logger.js';
import {BaseTool} from './base-tool.js';

const STAGES: Record<string, string> = {
    ir: 'ir',
    'virtual instructions': 'vinst',
    ast: 'ast',
};
const DEFAULT_STAGE = 'ir';

// ECPPS options start with '/', so it reads an absolute path as an unknown option; the tool runs in the source's
// directory, so pass the file relative to it. (Same as `ecppsInputArgument` in lib/compilers/ecpps.ts; importing
// that from here would create an import cycle through base-compiler and the options handler.)
function ecppsInputArgument(inputFilepath: string): string {
    return './' + path.basename(inputFilepath);
}

// Options the tool sets itself, so copies from the compiler or user options are dropped.
const OWN_OPTION = /^\/(verbose(:.*)?|asm|out:.*|output:.*)$/i;

export class EcppsIrTool extends BaseTool {
    static get key() {
        return 'ecpps-ir-tool';
    }

    static stageFor(args?: string[]): string {
        const wanted = (args ?? []).join(' ').trim().toLowerCase();
        return STAGES[wanted] ?? DEFAULT_STAGE;
    }

    static buildArguments(
        stage: string,
        outputFilepath: string,
        inputFilepath: string,
        compilerOptions: string[],
        userOptions: string[],
    ): string[] {
        const inputs = new Set([inputFilepath, ecppsInputArgument(inputFilepath)]);
        const keep = (option: string) => !OWN_OPTION.test(option) && !inputs.has(option);
        return [
            `/verbose:${stage}`,
            '/asm',
            `/out:${outputFilepath}`,
            ...compilerOptions.filter(keep),
            ...userOptions.filter(keep),
            ecppsInputArgument(inputFilepath),
        ];
    }

    override async runTool(compilationInfo: CompilationInfo, inputFilepath?: string, args?: string[]) {
        if (!inputFilepath) return this.createErrorResponse('No input file');

        const stage = EcppsIrTool.stageFor(args);
        const dirPath = path.dirname(inputFilepath);
        const outputFilepath = path.join(dirPath, `ecpps-${stage}.txt`);
        // Like BaseTool.runTool: the tool's own configured options come first.
        const compilerOptions = [
            ...this.tool.options,
            ...(compilationInfo.compiler.options ? splitArguments(compilationInfo.compiler.options) : []),
        ];
        const toolArgs = EcppsIrTool.buildArguments(
            stage,
            outputFilepath,
            inputFilepath,
            compilerOptions,
            compilationInfo.options ?? [],
        );

        const execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = dirPath;

        try {
            const result = await this.exec(compilationInfo.compiler.exe, toolArgs, execOptions);
            // When compilation fails the dump is empty or missing, and the diagnostics are in the tool's output.
            const dump = await fs.readFile(outputFilepath, 'utf8').catch(() => '');
            const transformedInput = result.filenameTransform(inputFilepath);
            const stdout = dump
                ? dump
                      .replace(/\r?\n$/, '')
                      .split(/\r?\n/)
                      .map(text => ({text}))
                : this.parseOutput(result.stdout, transformedInput);
            const toolResult: ToolResult = {
                id: this.tool.id,
                name: this.tool.name,
                code: result.code,
                languageId: this.tool.languageId,
                stdout,
                stderr: this.parseOutput(result.stderr, transformedInput, path.dirname(compilationInfo.compiler.exe)),
            };
            return toolResult;
        } catch (e) {
            logger.error('Error while running ECPPS IR tool: ', e);
            return this.createErrorResponse('Error while running tool');
        }
    }
}
