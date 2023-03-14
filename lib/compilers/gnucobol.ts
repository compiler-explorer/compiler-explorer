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

import _ from 'underscore';

import path from 'path';

import type {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import * as utils from '../utils.js';
import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class GnuCobolCompiler extends BaseCompiler {
    private readonly configDir: string;
    private readonly copyDir: string;
    private readonly includeDir: string;
    private readonly libDir: string;

    constructor(compilerInfo: PreliminaryCompilerInfo & {disabledFilters?: string[]}, env: CompilationEnvironment) {
        super(compilerInfo, env);
        const root = path.resolve(path.join(path.dirname(this.compiler.exe), '..'));
        this.includeDir = path.join(root, 'include');
        this.libDir = path.join(root, 'lib');
        this.configDir = path.join(root, 'share/gnucobol/config');
        this.copyDir = path.join(root, 'share/gnucobol/copy');
    }

    static get key() {
        return 'gnucobol';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        let options = ['-o', this.filename(outputFilename), '-I', this.includeDir, '-L', this.libDir];
        if (!filters.binary) options = options.concat('-S');
        else options = options.concat('-x');
        return options;
    }

    override getCompilerResultLanguageId() {
        return 'asm';
    }

    override getDefaultExecOptions(): ExecutionOptions {
        const result = super.getDefaultExecOptions();
        result.env.COB_CONFIG_DIR = this.configDir;
        result.env.COB_COPY_DIR = this.copyDir;
        return result;
    }

    override async objdump(
        outputFilename,
        result: any,
        maxSize: number,
        intelAsm,
        demangle,
        staticReloc: boolean,
        dynamicReloc: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const objdumpResult = await super.objdump(
            outputFilename,
            result,
            maxSize,
            intelAsm,
            demangle,
            staticReloc,
            dynamicReloc,
            filters,
        );

        objdumpResult.languageId = 'asm';
        return objdumpResult;
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, outputFilebase);
    }

    override getSharedLibraryPathsAsArguments(libraries, libDownloadPath) {
        return [];
    }

    override getSharedLibraryLinks(libraries: any[]): string[] {
        return [];
    }
}
