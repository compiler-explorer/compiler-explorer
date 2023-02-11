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

import fs from 'fs-extra';
import path from 'path';

import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import * as exec from '../exec.js';
import _ from 'underscore';

import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';

import type {ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';

import * as utils from '../utils.js';
import {logger} from '../logger.js';

// Although --redirect-code-traces=<filename> flag of v8 says it can write the output to
// any filename, I have not been able to make it do that. Instead, the output always
// goes to code.asm.
const v8AsmRedirectTargetFileName = 'code.asm';

export class V8Compiler extends BaseCompiler {
    static get key() {
        return 'v8';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
    }

    override getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        return this.filename(path.join(path.dirname(inputFilename), v8AsmRedirectTargetFileName));
    }

    public override getOutputFilename(dirPath: string, outputFilebase: string, key?: any) {
        return v8AsmRedirectTargetFileName;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return [];
    }

    override async doBuildstep(command, args, execParams) {
        delete execParams.env;
        delete execParams.ldPath;
        let result = await this.exec('cmake', ['-S', '..', '-B', '..'], execParams);
        result = await this.exec('cmake', ['--build', '..'], execParams);
        return this.processExecutionResult(result);
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string, key?) {
        let entryFile = 'example.mjs';
        // Pick filename from the cmake source
        const filenameVariableRegex = /set\(JAVASCRIPT_ENTRY_FILE(.*)\)/;
        if (filenameVariableRegex.test(key.source)) {
            const matches = key.source.match(filenameVariableRegex);
            if (matches && matches[1]) {
                entryFile = matches[1];
                entryFile = entryFile.replace(/['"]/g, '').trim();
            }
        }

        if (dirPath) {
            return path.join(dirPath, entryFile);
        } else {
            return entryFile;
        }
    }

    override async cmake(files, key) {
        // key = {source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries};

        _.defaults(key.filters, this.getDefaultFilters());
        key.filters.binary = true;
        key.filters.dontMaskFilenames = true;

        const libsAndOptions = this.createLibsAndOptions(key);

        const executeParameters: ExecutableExecutionOptions = {
            ldPath: this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries),
            args: key.executionParameters.args || [],
            stdin: key.executionParameters.stdin || '',
            env: {},
        };

        const cacheKey = this.getCmakeCacheKey(key, files);

        const dirPath = await this.newTempDir();

        const executableFileName = this.getExecutableFilename(path.join(dirPath, 'build'), this.outputFilebase, key);

        let fullResult = await this.loadPackageWithExecutable(cacheKey, dirPath);
        if (fullResult) {
            fullResult.fetchedFromCache = true;

            delete fullResult.inputFilename;
            delete fullResult.executableFilename;
            delete fullResult.dirPath;
        } else {
            let writeSummary;
            try {
                writeSummary = await this.writeAllFilesCMake(dirPath, cacheKey.source, files, cacheKey.filters);
            } catch (e) {
                return this.handleUserError(e, dirPath);
            }

            const execParams = this.getDefaultExecOptions();
            execParams.appHome = dirPath;
            execParams.customCwd = path.join(dirPath, 'build');

            await fs.mkdir(execParams.customCwd);

            const makeExecParams = this.createCmakeExecParams(execParams, dirPath, libsAndOptions);

            fullResult = {
                buildsteps: [],
                inputFilename: writeSummary.inputFilename,
            };

            fullResult.downloads = await this.setupBuildEnvironment(cacheKey, dirPath, true);

            const toolchainparam = this.getCMakeExtToolchainParam();

            const cmakeArgs = utils.splitArguments(key.backendOptions.cmakeArgs);
            const fullArgs: string[] = [toolchainparam, ...this.getExtraCMakeArgs(key), ...cmakeArgs, '..'];

            const cmakeStepResult = await this.doBuildstepAndAddToResult(
                fullResult,
                'cmake',
                this.env.ceProps('cmake'),
                fullArgs,
                makeExecParams,
            );

            if (cmakeStepResult.code !== 0) {
                fullResult.result = {
                    dirPath,
                    okToCache: false,
                    code: cmakeStepResult.code,
                    asm: [{text: '<Build failed>'}],
                };
                return fullResult;
            }

            fullResult.result = {
                dirPath,
                okToCache: true,
            };

            fullResult.result.compilationOptions = [];

            fullResult.code = 0;
            fullResult.code = fullResult.buildsteps.reduce((codesum, step) => {
                codesum += step.code;
                return codesum;
            }, 0);

            await this.storePackageWithExecutable(cacheKey, dirPath, fullResult);
        }

        fullResult.result.dirPath = dirPath;

        const maxExecOutputSize = this.env.ceProps('max-executable-output-size', 32 * 1024);
        const compilerArgs = this.compiler.options.split(' ').map(arg => arg.trim());
        fullResult.execResult = await this.execBinary(
            this.compiler.exe,
            maxExecOutputSize,
            {
                args: compilerArgs.concat(executableFileName),
                stdin: '',
                ldPath: [],
                env: {
                    ASAN_OPTIONS: 'color=always',
                    UBSAN_OPTIONS: 'color=always',
                    MSAN_OPTIONS: 'color=always',
                    LSAN_OPTIONS: 'color=always',
                },
            },
            dirPath,
        );

        fullResult.didExecute = true;

        const asmFile = path.join(fullResult.result.dirPath, v8AsmRedirectTargetFileName);
        try {
            const asmFileContents = await fs.readFile(asmFile, 'utf8');
            fullResult.asm = asmFileContents.split('\n').map(text => ({text}));
        } catch (err) {
            fullResult.asm = [];
            logger.error('Error during reading assembly for v8: ', err);
        }

        delete fullResult.result.dirPath;

        return fullResult;
    }
}
