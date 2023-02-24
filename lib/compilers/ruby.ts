// Copyright (c) 2021, Compiler Explorer Authors
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

import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {resolvePathFromAppRoot} from '../utils';

import {BaseParser} from './argument-parsers';

export class RubyCompiler extends BaseCompiler {
    disasmScriptPath: any;

    static get key() {
        return 'ruby';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        this.disasmScriptPath =
            this.compilerProps('disasmScript') || resolvePathFromAppRoot('etc', 'scripts', 'disasms', 'disasm.rb');
    }

    override getCompilerResultLanguageId() {
        return 'asmruby';
    }

    override processAsm(result) {
        const lineRe = /\(\s*(\d+)\)(?:\[[^\]]+])?$/;
        const fileRe = /ISeq:.*?@(.*?):(\d+) /;
        const baseFile = path.basename(this.compileFilename);

        const bytecodeLines = result.asm.split('\n');

        const bytecodeResult: any = [];
        let lastFile: any = null;
        let lastLineNo: any = null;

        for (const line of bytecodeLines) {
            const match = line.match(lineRe);

            if (match) {
                lastLineNo = parseInt(match[1]);
            } else if (line) {
                const fileMatch = line.match(fileRe);
                if (fileMatch) {
                    lastFile = fileMatch[1];
                    lastLineNo = parseInt(fileMatch[2]);
                }
            } else {
                lastFile = null;
                lastLineNo = null;
            }

            const file = lastFile === baseFile ? null : lastFile;
            const result = {text: line, source: {line: lastLineNo, file: file}};
            bytecodeResult.push(result);
        }

        return {asm: bytecodeResult};
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return [
            this.disasmScriptPath,
            '--outputfile',
            outputFilename,
            '--fname',
            path.basename(this.compileFilename),
            '--inputfile',
        ];
    }

    override getArgumentParser() {
        return BaseParser;
    }
}
