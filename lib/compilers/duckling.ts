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

import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {filterEscapeSequences} from '../utils.js';

export class DucklingCompiler extends BaseCompiler {
    static get key() {
        return 'duckling';
    }

    constructor(...args: ConstructorParameters<typeof BaseCompiler>) {
        super(...args);
        this.outputFilebase = 'example';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, _outputFilename: string): string[] {
        if (filters.binary || filters.execute) return ['compile_modules', '--dump-ir', 'asm'];
        return ['compile_module', '--dump-ir', 'asm'];
    }

    override getExecutableFilename(dirPath: string, _outputFilebase: string): string {
        return path.join(dirPath, 'duck_build', 'package_llvm.exe');
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

    private tagDiagnostics(lines: ResultLine[], inputFilename: string): void {
        const locationPattern = /^\s*>\s*(?<filename>.+):(?<line>\d+):(?<column>\d+)\s*$/;
        const diagnosticPattern = /^\s*(?<severity>error|warning|note|help)(?:\[[^\]]+\])?:/i;

        for (let index = 1; index < lines.length; index++) {
            const location = filterEscapeSequences(lines[index].text).match(locationPattern);
            if (!location?.groups) continue;

            const headline = lines[index - 1];
            const diagnosticText = filterEscapeSequences(headline.text);
            const diagnostic = diagnosticText.match(diagnosticPattern);
            if (!diagnostic?.groups) continue;

            const sourceFilename = location.groups.filename;
            const file =
                sourceFilename === '<source>'
                    ? path.basename(inputFilename)
                    : path
                          .relative(
                              path.dirname(inputFilename),
                              path.resolve(path.dirname(inputFilename), sourceFilename),
                          )
                          .split(path.sep)
                          .join('/');
            const line = Number.parseInt(location.groups.line, 10);
            const column = Number.parseInt(location.groups.column, 10);
            const severity = diagnostic.groups.severity.toLowerCase();

            headline.tag = {
                file,
                line,
                column,
                text: diagnosticText.trim(),
                severity: severity === 'warning' ? 2 : severity === 'note' || severity === 'help' ? 1 : 3,
            };
            lines[index].tag = {file, line, column, text: '', severity: 3};
        }
    }

    protected async getModuleSourceFiles(dirPath: string): Promise<string[]> {
        const files: string[] = [];

        const visitDirectory = async (currentPath: string): Promise<void> => {
            for (const entry of await fs.readdir(currentPath, {withFileTypes: true})) {
                const filename = path.join(currentPath, entry.name);
                if (entry.isDirectory()) {
                    if (entry.name !== 'duck_build' && !entry.name.startsWith('.')) {
                        await visitDirectory(filename);
                    }
                } else if (entry.isFile() && path.extname(entry.name) === '.dk') {
                    files.push(filename);
                }
            }
        };

        await visitDirectory(dirPath);
        return files.sort();
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

        const compilerOptions = [...options];
        if (compilerOptions[0] === 'compile_module' || compilerOptions[0] === 'compile_modules') {
            const moduleFiles = await this.getModuleSourceFiles(workingDirectory);
            const otherModuleFiles = moduleFiles
                .filter(filename => path.resolve(filename) !== path.resolve(inputFilename))
                .map(filename => this.filename(filename));

            if (otherModuleFiles.length > 0) {
                compilerOptions[0] = 'compile_modules';
                compilerOptions.splice(2, 0, ...otherModuleFiles);
                if (!filters?.execute && !filters?.binary) compilerOptions.push('--no-executable');
                if (!compilerOptions.includes('--no-incremental')) compilerOptions.push('--no-incremental');
            }
        }

        const result = await super.runCompiler(compiler, compilerOptions, inputFilename, execOptions, filters);
        this.tagDiagnostics(result.stderr, inputFilename);
        this.tagDiagnostics(result.stdout, inputFilename);
        if (result.code !== 0) return result;

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
                    ? `Duckling produced no assembly for ${path.basename(inputFilename)}`
                    : `Duckling produced multiple assembly files for ${path.basename(inputFilename)}`;
            result.stderr.push({text: message});
            return result;
        }

        const outputFilename = this.getOutputFilename(workingDirectory, this.outputFilebase);
        await fs.copyFile(path.join(artifactsDirectory, assemblyFiles[0]), outputFilename);
        return result;
    }
}
