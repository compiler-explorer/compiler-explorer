// Copyright (c) 2018, 2023, Elliot Saba & Compiler Explorer Authors
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

import {ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces';
import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import * as utils from '../utils';

import {JuliaParser} from './argument-parsers';

export class JuliaCompiler extends BaseCompiler {
    private compilerWrapperPath: string;

    static get key() {
        return 'julia';
    }

    constructor(info: PreliminaryCompilerInfo, env) {
        super(info, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper', '') ||
            utils.resolvePathFromAppRoot('etc', 'scripts', 'julia_wrapper.jl');
    }

    // No demangling for now
    override postProcessAsm(result) {
        return result;
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override processAsm(result, filters, options) {
        const lineRe = /^<(\d+) (\d+) ([^ ]+) ([^>]*)>$/;
        const bytecodeLines = result.asm.split('\n');
        const bytecodeResult: ParsedAsmResultLine[] = [];
        // Every method block starts with a introductory line
        //   <[source code line] [output line number] [function name] [method types]>
        // Check for the starting line, add the method block, skip other lines
        let i = 0;
        while (i < bytecodeLines.length) {
            const line = bytecodeLines[i];
            const match = line.match(lineRe);

            if (match) {
                const source = parseInt(match[1]);
                let linenum = parseInt(match[2]);
                linenum = Math.min(linenum, bytecodeLines.length);
                const funname = match[3];
                const types = match[4];
                let j = 0;
                bytecodeResult.push({text: '<' + funname + ' ' + types + '>', source: {line: source, file: null}});
                while (j < linenum) {
                    bytecodeResult.push({text: bytecodeLines[i + 1 + j], source: {line: source, file: null}});
                    j++;
                }
                bytecodeResult.push({text: '', source: {file: null}});
                i += linenum + 1;
                continue;
            }
            i++;
        }
        return {asm: bytecodeResult};
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return [];
    }

    override getArgumentParser() {
        return JuliaParser;
    }

    override fixExecuteParametersForInterpreting(executeParameters, outputFilename, key) {
        super.fixExecuteParametersForInterpreting(executeParameters, outputFilename, key);
        executeParameters.args.unshift('--');
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const dirPath = path.dirname(inputFilename);

        if (!execOptions.customCwd) {
            execOptions.customCwd = dirPath;
        }

        const juliaOptions = [this.compilerWrapperPath, '--'];
        options.push(this.getOutputFilename(dirPath, this.outputFilebase));
        juliaOptions.push(...options);

        const execResult = await this.exec(compiler, juliaOptions, execOptions);
        return {
            compilationOptions: juliaOptions,
            ...this.transformToCompilationResult(execResult, inputFilename),
        };
    }
}
