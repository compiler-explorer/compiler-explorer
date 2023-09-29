// Copyright (c) 2018, Forschungzentrum Juelich GmbH, Juelich Supercomputing Centre
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

import type {
    CompilationResult,
    CompileChildLibraries,
    ExecutionOptions,
} from '../../types/compilation/compilation.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import * as utils from '../utils.js';
import {GccFortranParser} from './argument-parsers.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import _ from 'underscore';

export class FortranCompiler extends BaseCompiler {
    static get key() {
        return 'fortran';
    }

    protected override getArgumentParser(): any {
        return GccFortranParser;
    }

    override getStdVerOverrideDescription(): string {
        return 'Change the Fortran standard version of the compiler.';
    }

    override getSharedLibraryPaths(libraries: CompileChildLibraries[]) {
        return libraries
            .map(selectedLib => {
                const foundVersion = this.findLibVersion(selectedLib);
                if (!foundVersion) return false;

                const paths = [...foundVersion.libpath];
                if (this.buildenvsetup && !this.buildenvsetup.extractAllToRoot) {
                    paths.push(`/app/${selectedLib.id}/lib`);
                }
                return paths;
            })
            .flat();
    }

    override getIncludeArguments(libraries: SelectedLibraryVersion[]): string[] {
        const includeFlag = this.compiler.includeFlag || '-I';
        return libraries.flatMap(selectedLib => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return [];

            const paths = foundVersion.path.map(path => includeFlag + path);
            if (foundVersion.packagedheaders) {
                paths.push(`-I/app/${selectedLib.id}/mod`);
                paths.push(includeFlag + `/app/${selectedLib.id}/include`);
            }
            return paths;
        });
    }

    override getSharedLibraryPathsAsArguments(
        libraries: CompileChildLibraries[],
        libDownloadPath?: string,
        toolchainPath?: string,
    ) {
        const pathFlag = this.compiler.rpathFlag || this.defaultRpathFlag;
        const libPathFlag = this.compiler.libpathFlag || '-L';

        let toolchainLibraryPaths: string[] = [];
        if (toolchainPath) {
            toolchainLibraryPaths = [path.join(toolchainPath, '/lib64'), path.join(toolchainPath, '/lib32')];
        }

        if (!libDownloadPath) {
            libDownloadPath = './lib';
        }

        return _.union(
            [libPathFlag + libDownloadPath],
            [pathFlag + libDownloadPath],
            this.compiler.libPath.map(path => pathFlag + path),
            toolchainLibraryPaths.map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries).map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries).map(path => libPathFlag + path),
        ) as string[];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions & {env: Record<string, string>},
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }
        // Switch working directory of compiler to temp directory that also holds the source.
        // This makes it possible to generate .mod files.
        execOptions.customCwd = path.dirname(inputFilename);

        const result = await this.exec(compiler, options, execOptions);
        const baseFilename = './' + path.basename(inputFilename);
        return {
            ...result,
            stdout: utils.parseOutput(result.stdout, baseFilename),
            stderr: utils.parseOutput(result.stderr, baseFilename),
            inputFilename: inputFilename,
        };
    }
}
