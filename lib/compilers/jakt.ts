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

import {ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';

export class JaktCompiler extends BaseCompiler {
    static get key() {
        return 'jakt';
    }

    constructor(info, env) {
        super(info, env);

        this.outputFilebase = 'example';
    }

    override getCompilerResultLanguageId() {
        return 'cppp';
    }

    override async objdump(
        outputFilename,
        result: any,
        maxSize: number,
        intelAsm,
        demangle,
        filters: ParseFiltersAndOutputOptions,
    ) {
        const objdumpResult = await super.objdump(outputFilename, result, maxSize, intelAsm, demangle, filters);
        objdumpResult.languageId = 'asm';
        return objdumpResult;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: any) {
        return ['--binary-dir', path.dirname(outputFilename)];
    }

    override getObjdumpOutputFilename(defaultOutputFilename) {
        const parsed_path = path.parse(defaultOutputFilename);

        return path.join(parsed_path.dir, this.outputFilebase);
    }

    override getExecutableFilename(dirPath, outputFilebase, key?) {
        return path.join(dirPath, outputFilebase);
    }

    // We have no dynamic linking in Jakt
    override getSharedLibraryPathsAsArguments(libraries, libDownloadPath) {
        return [];
    }

    // We have no dynamic linking in Jakt
    override getSharedLibraryLinks(libraries): string[] {
        return [];
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: any): string {
        return path.join(dirPath, 'Root Module.cpp');
    }
}
