// Copyright (c) 2024-2025, Michele Martone and Compiler Explorer Authors
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

import {open} from 'node:fs/promises';
import path from 'node:path';

import {CacheKey, CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {BasicExecutionResult, UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import * as utils from '../utils.js';

export class CoccinelleCCompiler extends BaseCompiler {
    protected spatchBaseFilename: string; // if true, doTempfolderCleanup won't clean up
    protected joinSpatchStdinAndStderr: boolean; // dirty hopefullytemporary hack, as coccinelle dumps diagnostics on both streams
    protected verbose: boolean; // for maintenance/development
    protected permittedOptions: string[];

    static get key() {
        return 'coccinelle_for_c';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(
            {
                disabledFilters: ['labels', 'directives', 'commentOnly', 'trim', 'debugCalls'],
                ...info,
            },
            env,
        );
        this.compiler.supportsIntel = false;
        this.delayCleanupTemp = false;
        this.spatchBaseFilename = 'patch.cocci';
        this.outputFilebase = 'output';
        this.joinSpatchStdinAndStderr = true;
        this.verbose = false;
        this.permittedOptions = ['--debug', '--help', '--iso-limit=0', '--quiet', '--verbose', '--very-quiet.'];
        // ...
    }

    override getDefaultFilters() {
        // We want to keep the entire output of Coccinelle (so, no filter)
        return {
            intel: false,
            commentOnly: false,
            directives: false,
            labels: false,
            optOutput: false,
            binary: false,
            execute: false,
            demangle: false,
            libraryCode: false,
            trim: false,
            binaryObject: false,
            debugCalls: false,
        };
    }

    override async populatePossibleOverrides() {
        await super.populatePossibleOverrides();
    }

    override getSharedLibraryPathsAsArguments(libraries: SelectedLibraryVersion[], libDownloadPath?: string) {
        return [];
    }

    override getSharedLibraryLinks(libraries: any[]): string[] {
        return [];
    }

    override getIncludeArguments(libraries: SelectedLibraryVersion[], dirPath: string): string[] {
        return super.getIncludeArguments(libraries, dirPath);
    }

    override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        return options.concat(userOptions, libIncludes, libOptions, libPaths, libLinks, staticLibLinks, [
            this.filename(inputFilename),
        ]);
    }

    override optionsForBackend(backendOptions: Record<string, any>, outputFilename: string) {
        return super.optionsForBackend(backendOptions, outputFilename);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        // coccinelle_for_c
        return ['--sp-file', this.spatchBaseFilename, '-o', this.filename(outputFilename)];
    }

    override optionsForDemangler(filters?: ParseFiltersAndOutputOptions): string[] {
        const options = super.optionsForDemangler(filters);
        if (filters !== undefined && !filters.verboseDemangling) {
            options.push('--no-verbose');
        }
        return options;
    }

    override getIrOutputFilename(inputFilename: string, filters: ParseFiltersAndOutputOptions): string {
        return super.getIrOutputFilename(inputFilename, filters);
    }

    override getArgumentParserClass() {
        return super.getArgumentParserClass();
    }

    override isCfgCompiler() {
        return false;
    }

    override processExecutionResult(input: UnprocessedExecResult, inputFilename?: string): BasicExecutionResult {
        return utils.processExecutionResult(input, inputFilename);
    }

    override buildExecutable(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        // may move here the file separation logic currently in runCompiler
        return super.buildExecutable(compiler, options, inputFilename, execOptions);
    }

    checkCocciSource(source: string) {
        // forbid '#spatch'-alike lines anywhere in the semantic patch
        const re = /^\s*#\s*spatch.*/;
        const failed: string[] = [];
        for (const [index, line] of utils.splitLines(source).entries()) {
            if (re.test(line)) {
                failed.push(`<stdin>:${index + 1}:1: no #spatch lines please`);
            }
        }
        if (failed.length > 0) return failed.join('\n');
        return null;
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        // here to extract contents of source and semantic patch
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const spatchFilename = path.join(path.dirname(inputFilename), this.spatchBaseFilename);
        let cFileContents = '';
        let pFileContents = '';
        let toc = true; // toc: Target Language or Coccinelle Language
        const file = await open(inputFilename);
        for await (const line of file.readLines()) {
            // this separates the C/C++ portion from the SmPL
            const ifdefCocciRegex = /^#ifdef.*COCCINELLE.*/;
            const endifCocciRegex = /^#endif.*COCCINELLE.*/;

            if (ifdefCocciRegex.test(line) && toc) {
                toc = !toc;
            } else {
                if (!endifCocciRegex.test(line)) {
                    if (toc) cFileContents += line + '\n';
                    else pFileContents += line + '\n';
                }
            }
        }
        file.close();

        // we overwrite the C/C++ + SmPL with extracted C/C++ sources (we also spare ourselves an unlink())
        const cfile = await open(inputFilename, 'w');
        cfile.write(cFileContents);
        cfile.close();

        const spfile = await open(spatchFilename, 'w');
        // we save the extracted SmPL (semantic patch) sources
        spfile.write(pFileContents);
        spfile.close();

        // check that the SmPL source does not #include anything, via CE rules
        let cocciSourceError = this.checkSource(pFileContents);
        if (cocciSourceError) throw cocciSourceError;
        // use own rules, too
        cocciSourceError = this.checkCocciSource(pFileContents);
        if (cocciSourceError) throw cocciSourceError;

        const result = await this.exec(compiler, options, execOptions);

        if (this.joinSpatchStdinAndStderr && result.code === 0 && !result.timedOut) {
            for (const opt of options)
                if (opt == '--help')
                    result.stdout +=
                        '\n**** Of the above, only the following are allowed here in Compiler Explorer: ****\n' +
                        this.permittedOptions.concat('') +
                        '\n';
            result.stderr += result.stdout;
            result.stdout = result.stderr;
            result.stderr = '';
        }
        return {
            ...this.transformToCompilationResult(result, inputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
            instructionSet: this.getInstructionSetFromCompilerArgs(options),
        };
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey): string {
        return path.join(dirPath, `${outputFilebase}${this.lang.extensions[0]}`);
    }

    override postCompilationPreCacheHook(result: CompilationResult): CompilationResult {
        if (result.code === 0) {
            result.languageId = 'c'; // we produce C
        }
        return result;
    }

    override filterUserOptions(args: string[]) {
        const permittedOptionsArray = new Set(this.permittedOptions);
        return args.filter(item => {
            if (typeof item !== 'string') return true;

            if (permittedOptionsArray.has(item)) return true;

            if (this.verbose) logger.warn(`User-provided option ${item} not allowed -- ignoring it.`);
            return false;
        });
    }
}

export class CoccinelleCPlusPlusCompiler extends CoccinelleCCompiler {
    static override get key() {
        return 'coccinelle_for_cpp';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        // coccinelle_for_cpp
        const options = super.optionsForFilter(filters, outputFilename, userOptions);
        options.push('--c++');
        return options;
    }

    override postCompilationPreCacheHook(result: CompilationResult): CompilationResult {
        if (result.code === 0) {
            result.languageId = 'cpp'; // we produce C++
        }
        return result;
    }
}
