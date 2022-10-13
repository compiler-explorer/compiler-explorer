// Copyright (c) 2019, Compiler Explorer Authors
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

import {ParseFilters} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {AsmRaw} from '../parsers/asm-raw';
import * as utils from '../utils';

export class Cc65Compiler extends BaseCompiler {
    static get key() {
        return 'cc65';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.asm = new AsmRaw();
        this.toolchainPath = path.resolve(path.dirname(compilerInfo.exe), '..');
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters, outputFilename) {
        return ['-g', '-o', this.filename(outputFilename)];
    }

    override async getCmakeBaseEnv() {
        if (!this.compiler.exe) return {};

        const env: Record<string, string> = {};

        env.CC = this.compiler.exe;

        if (this.toolchainPath) {
            const ldPath = `${this.toolchainPath}/bin/ld65`;
            const arPath = `${this.toolchainPath}/bin/ar65`;
            const asPath = `${this.toolchainPath}/bin/as65`;

            if (await utils.fileExists(ldPath)) env.LD = ldPath;
            if (await utils.fileExists(arPath)) env.AR = arPath;
            if (await utils.fileExists(asPath)) env.AS = asPath;
        }

        return env;
    }

    override async objdump(outputFilename, result: any, maxSize: number, intelAsm, demangle, filters: ParseFilters) {
        const res = await super.objdump(outputFilename, result, maxSize, intelAsm, demangle, filters);

        const dirPath = path.dirname(outputFilename);
        const nesFile = path.join(dirPath, 'example.nes');
        if (await utils.fileExists(nesFile)) {
            const file_buffer = await fs.readFile(nesFile);
            const binary_base64 = file_buffer.toString('base64');
            result.jsnesrom = binary_base64;
        }

        return res;
    }
}
