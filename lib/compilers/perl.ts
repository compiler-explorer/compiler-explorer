// Copyright (c) 2026, Compiler Explorer Authors
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
import {CompilationEnvironment} from '../compilation-env.js';
import {resolvePathFromAppRoot} from '../utils.js';

import {BaseParser} from './argument-parsers.js';

export class PerlCompiler extends BaseCompiler {
    private readonly disasmLibPath: string;
    private readonly disasmModule: string;

    static get key() {
        return 'perl';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compiler.demangler = '';
        this.demanglerClass = null;
        this.disasmLibPath =
            this.compilerProps<string>('disasmLibPath') || resolvePathFromAppRoot('etc', 'scripts', 'disasms');
        this.disasmModule = this.compilerProps<string>('disasmModule') || 'diswrapper';
    }

    override async processAsm(result) {
        // only nextstates have line numbers
        // state op parameters are (optlabel stash seq_no file:lineno)
        // I miss /x
        const nextstateRe =
            /^(?:-|\w+)\s+<;> (?:ex-)?(?:next|db)state\((?:[^:\s]+: )?[^:\s]+(?:::[^:\s]+)* \d+ ([^\s:]+):(\d+)\)/;
        const functionTopRe = /:$/;

        const bytecodeLines = result.asm.split('\n');

        const bytecodeResult: ParsedAsmResultLine[] = [];
        let lastLineNo: number | null = null;
        let sourceLoc: AsmResultSource | null = null;

        for (const line of bytecodeLines) {
            const match = line.match(nextstateRe);

            if (match) {
                const lineno = Number.parseInt(match[2], 10);
                sourceLoc = {line: lineno, file: null};
                lastLineNo = lineno;
            } else if (line && !line.match(functionTopRe)) {
                sourceLoc = {line: lastLineNo, file: null};
            } else {
                sourceLoc = {line: null, file: null};
                lastLineNo = null;
            }

            bytecodeResult.push({text: line, source: sourceLoc});
        }

        return {
            asm: bytecodeResult,
            languageId: 'perl-concise',
            asmKeywordTypes: ['keyword.perl-concise'],
        };
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['-I', this.disasmLibPath, '-M' + this.disasmModule + '=' + outputFilename, '-c'];
    }

    override getArgumentParserClass() {
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
