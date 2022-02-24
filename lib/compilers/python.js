// Copyright (c) 2019, Sebastian Rath
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

import { BaseCompiler } from '../base-compiler';
import { resolvePathFromAppRoot } from '../utils';

import { BaseParser } from './argument-parsers';

export class PythonCompiler extends BaseCompiler {
    static get key() { return 'python'; }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);
        this.compiler.demangler = null;
        this.demanglerClass = null;
        this.disasmScriptPath =
            this.compilerProps('disasmScript') || resolvePathFromAppRoot('etc', 'scripts', 'disasms', 'dis_all.py');
    }

    // eslint-disable-next-line no-unused-vars
    processAsm(result, filters) {
        const lineRe = /^\s{0,4}(\d+)(.*)/;

        const bytecodeLines = result.asm.split('\n');

        const bytecodeResult = [];
        let lastLineNo = null;
        let sourceLoc = null;

        for (const line of bytecodeLines) {
            const match = line.match(lineRe);

            if (match) {
                const lineno = parseInt(match[1]);
                sourceLoc = {line: lineno, file: null};
                lastLineNo = lineno;
            } else if (!line) {
                sourceLoc = {line: null, file: null};
                lastLineNo = null;
            } else {
                sourceLoc = {line: lastLineNo, file: null};
            }

            bytecodeResult.push({text: line, source: sourceLoc});
        }

        return {asm: bytecodeResult};
    }

    optionsForFilter(filters, outputFilename) {
        return ['-I',
            this.disasmScriptPath,
            '--outputfile',
            outputFilename,
            '--inputfile'];
    }

    getArgumentParser() {
        return BaseParser;
    }
}
