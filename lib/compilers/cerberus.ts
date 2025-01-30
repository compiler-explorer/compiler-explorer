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

import {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import {BypassCache, CacheKey, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {assert} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import * as utils from '../utils.js';

export class CerberusCompiler extends BaseCompiler {
    static get key() {
        return 'cerberus';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(
            {
                // Default is to disable all "cosmetic" filters
                disabledFilters: ['labels', 'directives', 'commentOnly', 'trim', 'debugCalls'],
                ...compilerInfo,
            },
            env,
        );
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFileName: string) {
        filters.binary = true;
        return ['-c', '-o', outputFileName];
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.co`);
    }

    override async objdump(outputFilename: string, result: any, maxSize: number) {
        if (!(await utils.fileExists(outputFilename))) {
            result.asm = '<No output file ' + outputFilename + '>';
            return result;
        }

        const execOptions: ExecutionOptions = {
            maxOutput: maxSize,
            customCwd: (result.dirPath as string) || path.dirname(outputFilename),
        };

        const args = ['--pp=core', outputFilename];

        const objResult = await this.exec(this.compiler.objdumper, args, execOptions);
        if (objResult.code === 0) {
            result.objdumpTime = objResult.execTime;
            result.asm = this.postProcessObjdumpOutput(objResult.stdout);
        } else {
            logger.error(`Error executing objdump ${this.compiler.objdumper}`, objResult);
            result.asm = `<No output: objdump returned ${objResult.code}>`;
        }

        return result;
    }

    override async handleInterpreting(key: CacheKey, executeParameters: ExecutableExecutionOptions) {
        const executionPackageHash = this.env.getExecutableHash(key);
        const compileResult = await this.getOrBuildExecutable(key, BypassCache.None, executionPackageHash);
        assert(compileResult.dirPath !== undefined);
        if (compileResult.code === 0) {
            executeParameters.args = [
                '--exec',
                this.getOutputFilename(compileResult.dirPath),
                '--',
                ...executeParameters.args,
            ];
            const result = await this.runExecutable(this.compiler.exe, executeParameters, compileResult.dirPath);
            return {
                ...result,
                didExecute: true,
                buildResult: compileResult,
            };
        } else {
            return {
                stdout: compileResult.stdout,
                stderr: compileResult.stderr,
                code: compileResult.code,
                didExecute: false,
                buildResult: compileResult,
                timedOut: false,
            };
        }
    }

    override async processAsm(result): Promise<ParsedAsmResult> {
        // Handle "error" documents.
        if (!result.asm.includes('\n') && result.asm[0] === '<') {
            return {asm: [{text: result.asm, source: null}]};
        }

        const lines = result.asm.split('\n');
        const plines = lines.map((l: string) => ({text: l}));
        return {
            asm: plines,
            languageId: 'core',
        };
    }
}
