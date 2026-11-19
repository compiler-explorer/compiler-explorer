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

import type {
    CacheKey,
    CompilationCacheKey,
    CompilationResult,
    ExecutionOptionsWithEnv,
} from '../../types/compilation/compilation.interfaces.js';
import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import type {ClientOptionsType} from '../options-handler.js';
import {type BaseParser, DucklingParser} from './argument-parsers.js';

export class DucklingCompiler extends BaseCompiler {
    static get key() {
        return 'duckling';
    }

    constructor(...args: ConstructorParameters<typeof BaseCompiler>) {
        super(...args);
        this.outputFilebase = 'example';
    }

    protected override getArgumentParserClass(): typeof BaseParser {
        return DucklingParser;
    }

    override async initialise(
        mtime: Date,
        clientOptions: ClientOptionsType,
        isPrediscovered = false,
    ): Promise<BaseCompiler | null> {
        const compiler = await super.initialise(mtime, clientOptions, isPrediscovered);
        if (compiler && !isPrediscovered && this.compiler.cachedPossibleArguments) {
            this.possibleArguments.populateOptions(this.compiler.cachedPossibleArguments);
            delete this.compiler.cachedPossibleArguments;
        }
        return compiler;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        if (filters.binary || filters.execute) {
            return ['compile_modules', '--no-incremental', '--output-file-name', path.basename(outputFilename)];
        }
        return filters.binaryObject
            ? ['compile_module', '--no-incremental']
            : ['compile_module', '--no-incremental', '--dump-ir', 'asm'];
    }

    protected override shouldGenerateIRFromMainCompilation(): boolean {
        return true;
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
        const options = super.prepareArguments(
            userOptions,
            filters,
            backendOptions,
            inputFilename,
            outputFilename,
            libraries,
            overrides,
        );
        const irOptions: LLVMIrBackendOptions | undefined = backendOptions?.produceIr;
        if (!irOptions || !this.compiler.supportsIrView || filters.execute) return options;

        const dumpCategories = new Set<string>();
        const args = options.slice(0, 2);
        for (let i = 2; i < options.length; i++) {
            const option = options[i];
            if (option === '--dump-ir' && options[i + 1] && !options[i + 1].startsWith('-')) {
                for (const category of options[++i].split(',')) dumpCategories.add(category);
            } else if (option.startsWith('--dump-ir=')) {
                for (const category of option.slice('--dump-ir='.length).split(',')) dumpCategories.add(category);
            } else {
                args.push(option);
            }
        }
        dumpCategories.add(irOptions.showOptimized ? 'llvm-opt' : 'llvm');
        return [...args, '--dump-ir', [...dumpCategories].join(',')];
    }

    override async processIrOutput(output: CompilationResult, irOptions: LLVMIrBackendOptions) {
        if (!output.inputFilename) throw new Error('Missing Duckling input filename');

        const inputFilename = path.basename(output.inputFilename);
        const moduleName = path.basename(inputFilename, path.extname(inputFilename));
        const artifactsDirectory = path.join(path.dirname(output.inputFilename), 'duck_build', 'duck_debug_artifacts');
        const extension = irOptions.showOptimized ? 'opt.ll' : 'll';
        const matches = (await fs.readdir(artifactsDirectory)).filter(filename =>
            filename.endsWith(`.${moduleName}.${extension}`),
        );
        if (matches.length !== 1) {
            throw new Error(`Expected one .${extension} artifact for ${inputFilename}, found ${matches.length}`);
        }

        const ir = await fs.readFile(path.join(artifactsDirectory, matches[0]), 'utf8');
        return this.llvmIr.process(ir, irOptions);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey): string {
        return this.isCacheKey(key) && key.filters?.binary
            ? this.getExecutableFilename(dirPath, outputFilebase)
            : super.getOutputFilename(dirPath, outputFilebase, key);
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string): string {
        return path.join(dirPath, 'duck_build', outputFilebase);
    }

    protected override getSharedLibraryPathsAsArguments(): string[] {
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
    ): string[] {
        return [
            ...options.slice(0, 1),
            this.filename(inputFilename),
            ...options.slice(1),
            ...libIncludes,
            ...libOptions,
            ...libPaths,
            ...libLinks,
            ...userOptions,
            ...staticLibLinks,
        ];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const workingDirectory = path.dirname(inputFilename);
        const duckBuildDirectory = path.join(workingDirectory, 'duck_build');
        await fs.rm(duckBuildDirectory, {recursive: true, force: true});
        await fs.mkdir(duckBuildDirectory, {recursive: true});

        execOptions ??= this.getDefaultExecOptions();
        execOptions.customCwd = workingDirectory;

        const result = await super.runCompiler(compiler, options, inputFilename, execOptions, filters);
        if (result.code !== 0 || filters?.binary || filters?.execute) return result;

        const outputFilename = this.getOutputFilename(workingDirectory, this.outputFilebase);
        if (filters?.binaryObject) {
            const objectFiles = (await fs.readdir(duckBuildDirectory, {recursive: true})).filter(filename =>
                filename.endsWith('.o'),
            );
            if (objectFiles.length !== 1) {
                result.code = 1;
                result.stderr.push({
                    text: `Duckling reported success but produced ${objectFiles.length} object files; expected exactly one`,
                });
                return result;
            }
            await fs.copyFile(path.join(duckBuildDirectory, objectFiles[0]), outputFilename);
            return result;
        }

        const moduleName = path.basename(inputFilename, path.extname(inputFilename));
        const artifactsDirectory = path.join(duckBuildDirectory, 'duck_debug_artifacts');
        let artifacts: string[] = [];
        try {
            artifacts = await fs.readdir(artifactsDirectory);
        } catch {
            // A successful compiler exit without this directory means there is no assembly to display.
        }

        const assemblyFiles = artifacts.filter(filename => filename.endsWith(`.${moduleName}.s`));
        if (assemblyFiles.length !== 1) {
            result.code = 1;
            const message =
                assemblyFiles.length === 0
                    ? `Duckling reported success but produced no assembly for ${path.basename(inputFilename)}`
                    : `Duckling produced multiple assembly files for ${path.basename(inputFilename)}`;
            result.stderr.push({text: message});
            return result;
        }

        await fs.copyFile(path.join(artifactsDirectory, assemblyFiles[0]), outputFilename);
        return result;
    }
}
