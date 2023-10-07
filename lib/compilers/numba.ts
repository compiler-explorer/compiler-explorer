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
import {CompilationEnvironment} from '../compilation-env.js';
import {AsmParser} from '../parsers/asm-parser.js';
import {resolvePathFromAppRoot} from '../utils.js';

import {BaseParser} from './argument-parsers.js';

export class NumbaCompiler extends BaseCompiler {
    private compilerWrapperPath: string;

    static get key() {
        return 'numba';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper', '') || resolvePathFromAppRoot('etc', 'scripts', 'numba_wrapper.py');
    }

    override async processAsm(result, filters, options) {
        const processed = await super.processAsm(result, filters, options);
        if (!(this.asm instanceof AsmParser)) return processed;
        for (const item of processed.asm) {
            const match = item.text.match(/;(\d+)$/);
            if (!match) continue;
            item.text = item.text.slice(0, match.index);
            if (this.asm.hasOpcode(item.text, false, false)) {
                item.source = {line: parseInt(match[1]), file: null};
            }
        }
        return processed;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        return ['-I', this.compilerWrapperPath, '--outputfile', outputFilename, '--inputfile'];
    }

    override getArgumentParser() {
        return BaseParser;
    }

    override async postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        result = await super.postProcessAsm(result, filters);
        for (const item of result.asm) {
            let line = item.text;
            // Numba includes long and noisy, abi tags.
            line = line.replace(/\[abi:\w+\]/g, '');
            // Numba's custom name mangling is, sadly, not invertible.
            // It 'escapes' symbols to valid Python identifiers in a "_%02x" format, so
            // we cannot perfectly demangle since users can write coinciding identifiers.
            // Python qualifies scoped function names with "<locals>"; since the risk
            // from "_3clocals_3e" collisions is small, we decode it.
            line = line.replace(/::_3clocals_3e::/g, '::<locals>::');
            // Numba's generator arguments have many escaped symbols.
            line = line.replace(/::next\(\w+_20generator_28\w+\)/, demangle_symbols);
            item.text = line;
        }
        return result;
    }
}

function demangle_symbols(text: string): string {
    // Numba escaped non-word ascii characters to "_%02x"-formatted strings.
    return text.replace(/_([a-f0-9]{2})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
}
