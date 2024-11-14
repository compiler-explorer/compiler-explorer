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

import path from 'path';

import fs from 'fs-extra';

import type {
    CompilationResult,
    ExecutionOptions,
    ExecutionOptionsWithEnv,
} from '../../types/compilation/compilation.interfaces.js';
import type {
    OptPipelineBackendOptions,
    OptPipelineOutput,
} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {RacketPassDumpParser} from '../parsers/racket-pass-dump-parser.js';

export class RacketCompiler extends BaseCompiler {
    private raco: string;
    private passDumpParser: RacketPassDumpParser;

    static get key() {
        return 'racket';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(
            {
                // Disable output filters, as they currently don't do anything
                disabledFilters: ['labels', 'directives', 'commentOnly', 'trim', 'debugCalls'],
                ...info,
            },
            env,
        );
        // Revise this if we add released versions of Racket 8.12 or later
        if (this.compiler.isNightly) {
            this.compiler.optPipeline = {
                groupName: 'Linklet',
                // Disable all options and filters, currently unsupported
                supportedOptions: [],
                supportedFilters: [],
                monacoLanguage: 'scheme',
            };
        }
        this.raco = this.compilerProps<string>(`compiler.${this.compiler.id}.raco`);
        this.passDumpParser = new RacketPassDumpParser(this.compilerProps);
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        // We currently always compile to bytecode first and then decompile.
        // Forcing `binary` on like this ensures `objdump` will be called for
        // the decompilation phase.
        filters.binary = true;

        return [];
    }

    override isCfgCompiler() {
        return false;
    }

    override supportsObjdump(): boolean {
        return true;
    }

    override getSharedLibraryPathsAsArguments(libraries: SelectedLibraryVersion[], libDownloadPath?: string): string[] {
        return [];
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
        // Move input file to end of options
        return options.concat(userOptions, libIncludes, libOptions, libPaths, libLinks, staticLibLinks, [
            this.filename(inputFilename),
        ]);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        // Compile to bytecode via `raco make`
        options = [...options];
        options.unshift('make');

        // Replace input filename (which may be different than the default)
        // as in pipeline mode below
        options.pop();
        options.push(inputFilename);

        const makeResult = await this.exec(this.raco, options, execOptions);

        return this.transformToCompilationResult(makeResult, inputFilename);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, 'compiled', `${this.compileFilename.replace('.', '_')}.zo`);
    }

    override async objdump(
        outputFilename: string,
        result: any,
        maxSize: number,
        intelAsm: boolean,
        demangle: boolean,
        staticReloc: boolean | undefined,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions,
    ): Promise<any> {
        // Decompile to assembly via `raco decompile` with `disassemble` package
        const execOptions: ExecutionOptions = {
            maxOutput: maxSize,
            customCwd: (result.dirPath as string) || path.dirname(outputFilename),
        };
        const decompileResult = await this.exec(this.raco, ['decompile', outputFilename], execOptions);

        if (decompileResult.code) {
            logger.error('Error decompiling via `raco decompile`', decompileResult);
            result.asm = `<No output: \`raco decompile\` returned ${decompileResult.code}>`;
        }

        result.objdumpTime = decompileResult.execTime;
        result.asm = this.postProcessObjdumpOutput(decompileResult.stdout);

        return result;
    }

    override async processAsm(result: any, filters: ParseFiltersAndOutputOptions, options: string[]) {
        // TODO: Process and highlight decompiled output
        return {
            asm: [{text: result.asm}],
        };
    }

    override async generateOptPipeline(
        inputFilename: string,
        options: string[],
        filters: ParseFiltersAndOutputOptions,
        optPipelineOptions: OptPipelineBackendOptions,
    ): Promise<OptPipelineOutput | undefined> {
        // Use a separate directory so this is not affected by the main
        // compilation (which races in parallel)
        const pipelineDir = await this.newTempDir();
        const inputFile = this.filename(inputFilename);
        const pipelineFile = path.join(pipelineDir, path.basename(inputFile));
        await fs.copyFile(inputFile, pipelineFile);

        const execOptions = this.getDefaultExecOptions();
        execOptions.maxOutput = 1024 * 1024 * 1024;

        // Dump various optimisation passes during compilation
        execOptions.env['PLT_LINKLET_SHOW_CP0'] = '1';
        execOptions.env['PLT_LINKLET_SHOW_PASSES'] = 'all';
        execOptions.env['PLT_LINKLET_SHOW_ASSEMBLY'] = '1';

        const compileStart = performance.now();
        const output = await this.runCompiler(this.compiler.exe, options, pipelineFile, execOptions);
        const compileEnd = performance.now();

        if (output.timedOut) {
            return {
                error: 'Invocation timed out',
                results: {},
                compileTime: output.execTime || compileEnd - compileStart,
            };
        }

        if (output.truncated) {
            return {
                error: 'Exceeded max output limit',
                results: {},
                compileTime: output.execTime || compileEnd - compileStart,
            };
        }

        if (output.code !== 0) {
            return;
        }

        // Useful for local debugging
        // const passesFile = path.join(pipelineDir, 'passes.scm');
        // console.log('Passes:', passesFile);
        // await fs.writeFile(passesFile, output.stderr.map(l => l.text).join('\n'));

        try {
            const parseStart = performance.now();
            const llvmOptPipeline = await this.processOptPipeline(
                output,
                filters,
                optPipelineOptions,
                /* debugPatched = */ false,
            );
            const parseEnd = performance.now();

            return {
                results: llvmOptPipeline,
                compileTime: compileEnd - compileStart,
                parseTime: parseEnd - parseStart,
            };
        } catch (e: any) {
            return {
                error: e.toString(),
                results: {},
                compileTime: compileEnd - compileStart,
            };
        }
    }

    override async processOptPipeline(
        output: CompilationResult,
        filters: ParseFiltersAndOutputOptions,
        optPipelineOptions: OptPipelineBackendOptions,
        debugPatched?: boolean,
    ) {
        return this.passDumpParser.process(output.stderr, filters, optPipelineOptions);
    }
}
