// Copyright (c) 2025, Compiler Explorer Authors
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
        // Numba's function-end labels survive standard filtering.
        if (filters.labels) {
            processed.asm = processed.asm.filter(item => !item.text.startsWith('.Lfunc_end'));
        }
        if (!(this.asm instanceof AsmParser)) return processed;
        for (const item of processed.asm) {
            // We receive line numbers as comments to line ends.
            const match = item.text.match(/;(\d+)$/);
            if (!match) continue;
            item.text = item.text.slice(0, match.index);
            const inNvccCode = false;
            if (this.asm.hasOpcode(item.text, inNvccCode)) item.source = {line: Number.parseInt(match[1]), file: null};
        }
        return processed;
    }

    override async postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        result = await super.postProcessAsm(result, filters);
        for (const item of result.asm) {
            let line = item.text;
            // Numba includes long and noisy abi tags.
            line = line.replaceAll(/\[abi:\w+]/g, '');
            // Numba's custom name mangling is not invertible. It escapes symbols to
            // valid Python identifiers in a "_%02x" format. Because users can write
            // coinciding identifiers, we cannot perfectly demangle. Python qualifies
            // scoped function names with "<locals>". There is little risk from
            // collisions with user-defined symbols including `_3clocals_3e`.
            line = line.replaceAll('::_3clocals_3e::', '::<locals>::');
            // Numba's generators have many escaped symbols in their argument listings.
            line = line.replace(/::next\(\w+_20generator_28\w+\)/, decode_symbols);
            item.text = line;
        }
        return result;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        return ['-I', this.compilerWrapperPath, '--outputfile', outputFilename, '--inputfile'];
    }

    override getArgumentParserClass() {
        return BaseParser;
    }
}

export function decode_symbols(text: string): string {
    // Numba escapes /[^a-z0-9_]/ characters to "_%02x"-formatted strings.
    return text.replaceAll(/_([\da-f]{2})/g, (_, hex) => String.fromCodePoint(Number.parseInt(hex, 16)));
}
