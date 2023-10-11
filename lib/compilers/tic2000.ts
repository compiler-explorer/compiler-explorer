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

import {BaseCompiler} from '../base-compiler.js';
import {TiC2000AsmParser} from '../parsers/asm-parser-tic2000.js';

export class TIC2000 extends BaseCompiler {
    static get key() {
        return 'tic2000';
    }

    constructor(info, env) {
        super(info, env);
        this.outputFilebase = this.compileFilename.split('.')[0];
        this.asm = new TiC2000AsmParser(this.compilerProps);
    }

    override optionsForFilter(filters, outputFilename) {
        const options = ['-g', '-c', '-n', '--output_file=' + outputFilename];

        filters.preProcessLines = this.preProcessLines.bind(this);

        return options;
    }

    override getOutputFilename(dirPath, outputFilebase) {
        return path.join(dirPath, `${outputFilebase}.asm`);
    }

    preProcessLines(asmLines) {
        let i = 0;

        while (i < asmLines.length) {
            // Regex for determining the file line and column of the following source lines
            const match = asmLines[i].match(/^\s*\.dwpsn\s+file\s+(".*"),line\s+(\d+),column\s+(\d+)/);
            i++;
            if (match) {
                // Add two lines stating the file and location to allow parsing the source location by the standard
                // parser
                asmLines.splice(i, 0, '  .file 1 ' + match[1], '  .loc 1 ' + match[2] + ' ' + match[3]);
                i += 2;
            }
        }

        return asmLines;
    }
}
