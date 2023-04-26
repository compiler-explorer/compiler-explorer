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

import type {AsmResultSource, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {resolvePathFromAppRoot} from '../utils.js';

import {BaseParser} from './argument-parsers.js';

export class PythonCompiler extends BaseCompiler {
    private readonly disasmScriptPath: string;

    static get key() {
        return 'python';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
        this.disasmScriptPath =
            this.compilerProps<string>('disasmScript') ||
            resolvePathFromAppRoot('etc', 'scripts', 'disasms', 'dis_all.py');
    }

    override processAsm(result) {
        const lineRe = /^\s{0,4}(\d+)(.*)/;

        const bytecodeLines = result.asm.split('\n');

        const bytecodeResult: ParsedAsmResultLine[] = [];
        let lastLineNo: number | undefined;
        let sourceLoc: AsmResultSource | null = null;

        for (const line of bytecodeLines) {
            const match = line.match(lineRe);

            if (match) {
                const lineno = parseInt(match[1]);
                sourceLoc = {line: lineno, file: null};
                lastLineNo = lineno;
            } else if (line) {
                sourceLoc = {line: lastLineNo, file: null};
            } else {
                sourceLoc = {line: undefined, file: null};
                lastLineNo = undefined;
            }

            bytecodeResult.push({text: line, source: sourceLoc});
        }

        return {asm: bytecodeResult};
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['-I', this.disasmScriptPath, '--outputfile', outputFilename, '--inputfile'];
    }

    override getArgumentParser() {
        return BaseParser;
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
        return options.concat(
            [this.filename(inputFilename)],
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            userOptions,
            staticLibLinks,
        );
    }
}
