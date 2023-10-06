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

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {resolvePathFromAppRoot} from '../utils.js';

import {BaseParser} from './argument-parsers.js';

export class NumbaCompiler extends BaseCompiler {
    private readonly disasmScriptPath: string;

    static get key() {
        return 'numba';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        // TODO(Rupt) Add demangling / demanglerClass / demanglerType.
        this.disasmScriptPath =
            this.compilerProps<string>('disasmScript') || // TODO(Rupt) is this appropriate?
            resolvePathFromAppRoot('etc', 'scripts', 'numba_inspect.py');
    }

    override async processAsm(result, filters, options) {
        // TODO(Rupt) bug fix no line numbers if filtering comments
        const processed = await super.processAsm(result, filters, options);

        // TODO(Rupt) make magic comments more parsable!
        const magicCommentPattern = /^; CE_NUMBA (.+) (\(.*\)) (\d+)/;
        let lineno: number | undefined;

        for (const item of processed.asm) {
            const match = item.text.match(magicCommentPattern);
            if (match) {
                lineno = parseInt(match[3]);
                continue;
            }
            item.source = {line: lineno, file: null};
        }
        return processed;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        // TODO(Rupt): implement this for the numba script
        return ['-I', this.disasmScriptPath, '--outputfile', outputFilename, '--inputfile'];
    }

    override getArgumentParser() {
        return BaseParser;
    }
}
