// Copyright (c) 2022, Compiler Explorer Authors
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

import _ from 'underscore';

import { BaseCompiler } from '../base-compiler';

import { ClangParser } from './argument-parsers';

export class SolidityCompiler extends BaseCompiler {
    static get key() { return 'solidity'; }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    getArgumentParser() {
        return ClangParser;
    }

    optionsForFilter(filters, outputFilename, userOptions) {
        return [
            '--combined-json', 'asm', // We use it instead of `--asm-json` to have compacted json
            '-o', 'contracts',
        ];
    }

    isCfgCompiler(/*compilerVersion*/) {
        return false;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, 'contracts/combined.json');
    }

    processAsm(result) {
        // Handle "error" documents.
        if (!result.asm.includes('\n') && result.asm[0] === '<') {
            return {asm: [{text: result.asm}]};
        }

        return {
            asm: Object.entries(JSON.parse(result.asm).contracts)
                .sort(([_name1, data1], [_name2, data2]) =>
                    data1.asm['.code'][0].begin - data2.asm['.code'][0].begin,
                )
                .map(([name, data]) => {
                    const processOpcodes = (opcodes, indent) => opcodes
                        .map(opcode =>
                            `${opcode.name}${opcode.value !== undefined ? ` ${opcode.value}` : ''}`,
                        )
                        .map(opcode =>
                            (indent || '') + (opcode.startsWith('tag') ? opcode : `\t${opcode}`),
                        );

                    return [
                        `// ${_.last(name.split(':'))}`,
                        '.code',
                        processOpcodes(data.asm['.code']),
                        '.data',
                        Object.entries(data.asm['.data']).map(([id, {'.code': code}]) => [
                            `\t${id}:`,
                            '\t\t.code',
                            processOpcodes(code, '\t\t'),
                        ]),
                        '\n',
                    ];
                })
                .flat(Infinity)
                .slice(0, -1)
                .map(line => ({text: line})),
        };
    }
}
