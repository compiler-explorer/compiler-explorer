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
import _ from 'underscore';

import {CompilationResult} from '../../types/compilation/compilation.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import * as utils from '../utils';

import {ClangCompiler} from './clang';

export class LLVMMOSCompiler extends ClangCompiler {
    static override get key() {
        return 'llvmmos';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);
        this.externalparser = null;
        this.toolchainPath = path.normalize(path.join(path.dirname(this.compiler.exe), '..'));
    }

    override getExtraCMakeArgs(key): string[] {
        return [`-DCMAKE_PREFIX_PATH=${this.toolchainPath}`];
    }

    override fixFiltersBeforeCacheKey(filters, options, files) {
        filters.binary = false;
    }

    override getCMakeExtToolchainParam(): string {
        return '';
    }

    override async objdump(
        outputFilename,
        result: CompilationResult,
        maxSize: number,
        intelAsm,
        demangle,
        filters: ParseFiltersAndOutputOptions,
    ) {
        if (!outputFilename.endsWith('.elf') && (await utils.fileExists(outputFilename + '.elf'))) {
            outputFilename = outputFilename + '.elf';
        }

        intelAsm = false;
        const res = await super.objdump(outputFilename, result, maxSize, intelAsm, demangle, filters);

        if (this.compiler.exe.includes('nes')) {
            let nesFile = outputFilename;
            if (outputFilename.endsWith('.elf')) {
                nesFile = outputFilename.substr(0, outputFilename.length - 4);
            }

            if (await utils.fileExists(nesFile)) {
                const file_buffer = await fs.readFile(nesFile);
                const binary_base64 = file_buffer.toString('base64');
                result.jsnesrom = binary_base64;
            }
        }

        return res;
    }
}
