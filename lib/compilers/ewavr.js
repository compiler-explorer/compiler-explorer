// Copyright (c) 2019, Ethan Slattery
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

import temp from 'temp';

import { AsmEWAVRParser } from '../asm-parser-ewavr';
import { BaseCompiler } from '../base-compiler';

export class EWAVRCompiler extends BaseCompiler {
    static get key() { return 'ewavr'; }

    constructor(info, env) {
        info.supportsDemangle = false;
        info.supportsLibraryCodeFilter = false;
        super(info, env);
        this.asm = new AsmEWAVRParser(this.compilerProps);
    }

    newTempDir() {
        return new Promise((resolve, reject) => {
            temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.TMP}, (err, dirPath) => {
                if (err)
                    reject(`Unable to open temp file: ${err}`);
                else
                    resolve(dirPath);
            });
        });
    }

    optionsForFilter(filters, outputFilename) {
        if (filters.binary) {
            return [];
        }
        
        return [
            '-lB', this.filename(outputFilename),
            '-o', this.filename(outputFilename + '.obj'),
        ];
    }
}
