// Copyright (c) 2022, Compiler Explorer Authors
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

import type {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

import {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseParser} from './argument-parsers.js';

export class CarbonCompiler extends BaseCompiler {
    static get key() {
        return 'carbon';
    }

    override prepareArguments(
        userOptions: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        inputFilename: string,
        outputFilename: string,
        libraries: SelectedLibraryVersion[],
        overrides: ConfiguredOverrides,
    ): string[] {
        const args = super.prepareArguments(
            userOptions,
            filters,
            backendOptions,
            inputFilename,
            outputFilename,
            libraries,
            overrides,
        );
        // To support execution, we would need to work out how to run a separate "carbon link" stage
        // after generation. We have this with a couple of other compilers that can't do a compile-and-link in one step
        // and we've not yet got a good solution.
        args.unshift('compile');
        return args;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        const args = [`--output=${outputFilename}`];
        if (!filters.binary && !filters.binaryObject) args.push('--asm-output');
        return args;
    }
}

export class CarbonExplorerCompiler extends BaseCompiler {
    static get key() {
        return 'carbon-explorer';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        return ['--color', `--trace_file=${outputFilename}`];
    }

    override async processAsm(
        result,
        filters: ParseFiltersAndOutputOptions,
        options: string[],
    ): Promise<ParsedAsmResult> {
        // Really should write a custom parser, but for now just don't filter anything.
        return await super.processAsm(
            result,
            {
                labels: false,
                binary: false,
                commentOnly: false,
                demangle: false,
                optOutput: false,
                directives: false,
                dontMaskFilenames: false,
                execute: false,
                intel: false,
                libraryCode: false,
                trim: false,
                debugCalls: false,
            },
            options,
        );
    }

    lastLine(lines?: ResultLine[]): string {
        if (!lines || lines.length === 0) return '';
        return lines[lines.length - 1].text;
    }

    override postCompilationPreCacheHook(result: CompilationResult): CompilationResult {
        if (result.code === 0) {
            // Hook to parse out the "result: 123" line at the end of the interpreted execution run.
            const re = /^result: (\d+)$/;
            const match = re.exec(this.lastLine(result.asm as ResultLine[]));
            const code = match ? Number.parseInt(match[1]) : -1;
            result.execResult = {
                stdout: result.stdout,
                stderr: [],
                code: code,
                didExecute: true,
                timedOut: false,
                buildResult: {
                    code: 0,
                    timedOut: false,
                    stdout: [],
                    stderr: [],
                    downloads: [],
                    executableFilename: '',
                    compilationOptions: [],
                },
            };
            result.stdout = [];
            result.languageId = 'no-highlight';
        }
        return result;
    }

    override getArgumentParserClass() {
        // TODO: may need a custom one, based on/borrowing from ClangParser
        return BaseParser;
    }
}
