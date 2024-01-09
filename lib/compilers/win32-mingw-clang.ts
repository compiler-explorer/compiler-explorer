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

import {BuildResult, BypassCache, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';

import {copyNeededDlls} from '../win-utils.js';

import {ClangCompiler} from './clang.js';

export class Win32MingWClang extends ClangCompiler {
    static override get key() {
        return 'win32-mingw-clang';
    }

    override getExtraPaths(): string[] {
        const paths: string[] = super.getExtraPaths();

        return [...paths, path.normalize(path.dirname(this.compiler.exe))];
    }

    override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[],
    ): string[] {
        if (filters.binary) {
            filters.dontMaskFilenames = true;
        }

        return super.optionsForFilter(filters, outputFilename, userOptions);
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
        const newUserOptions = userOptions.filter(opt => !opt.startsWith('-l'));
        const newLinkOptions = userOptions.filter(opt => opt.startsWith('-l'));

        return options.concat(
            newUserOptions,
            [this.filename(inputFilename)],
            libIncludes,
            libOptions,
            libPaths,
            newLinkOptions,
            libLinks,
            staticLibLinks,
        );
    }

    override async buildExecutableInFolder(key, dirPath: string): Promise<BuildResult> {
        const result = await super.buildExecutableInFolder(key, dirPath);

        if (result.code === 0) {
            await copyNeededDlls(
                dirPath,
                result.executableFilename,
                this.exec,
                this.compiler.objdumper,
                this.getDefaultExecOptions(),
            );
        }

        return result;
    }

    override async handleExecution(key, executeParameters, bypassCache: BypassCache): Promise<CompilationResult> {
        const execOptions = this.getDefaultExecOptions();
        return super.handleExecution(key, {...executeParameters, env: execOptions.env}, bypassCache);
    }
}
