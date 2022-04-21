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

import * as fs from "fs";

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
            '--combined-json', 'asm,srcmap,srcmap-runtime,bin,bin-runtime,generated-sources-runtime', // We use it instead of `--asm-json` to have compacted json
            '-o', 'contracts',
        ];
    }

    isCfgCompiler(/*compilerVersion*/) {
        return false;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, 'contracts/combined.json');
    }

    parseSourceMap(sourceMapStr) {
        let prev = [0, 0, 0, 0, 0];
        return sourceMapStr.split(";").map((item) => {
            let ret = prev.map(x => x);

            item.split(":").forEach((value, index) => {
                if (value != "") {
                    ret[index] = value;
                }
            });

            prev = ret;
            return { start: ret[0], length: ret[1], sourceIndex: ret[2], jump: ret[3], modifierDepth: ret[4] };
        });
    }

    processAsm(result) {
        fs.writeFileSync("/home/jbr/out.json", JSON.stringify(result, null, 4));
        // Handle "error" documents.
        if (!result.asm.includes('\n') && result.asm[0] === '<') {
            return { asm: [{ text: result.asm }] };
        }

        const inputFile = fs.readFileSync(result.inputFilename);
        const inputFileStr = inputFile.toString();
        let currentLine = 1;
        const charToLine = inputFile.map(c => {
            const line = currentLine;
            if (c == "\n".charCodeAt(0)) {
                ++currentLine;
            }
            return line;
        });

        const asm = JSON.parse(result.asm);

        // TODO what happens if we have multiple contracts, can they share generated sources? probably not
        // TODO should lineToAsm be rebuilt per contract??

        let contracts = Object.entries(asm.contracts)
            .sort(([_name1, data1], [_name2, data2]) =>
                data1.asm['.code'][0].begin - data2.asm['.code'][0].begin,
            );

        return {
            asm: contracts.map(([name, data]) => {
                let generatedSources = {};
                // TODO check no overlapping regions, shouldn't be, but even so...
                data["generated-sources-runtime"].forEach(generatedSource => {
                    generatedSources[generatedSource.id] = generatedSource.ast.statements.map(statement => {
                        const srcSplit = statement.src.split(":").map(x => parseInt(x));
                        return { begin: srcSplit[0], end: srcSplit[0] + srcSplit[1], name: statement.name };
                    });
                });

                let lineToAsm = {};
                Object.entries(data.asm['.data']).forEach(([id, { '.code': code }]) => {
                    code.forEach((asmInstruction) => {
                        if (asmInstruction.source == 0) {
                            const firstLine = charToLine[asmInstruction.begin];
                            const lastLine = charToLine[asmInstruction.end];
                            for (let i = firstLine; i <= lastLine; ++i) {
                                if (lineToAsm[i] == undefined) {
                                    lineToAsm[i] = [];
                                }
                                lineToAsm[i].push(asmInstruction);
                                asmInstruction.line = i;
                            }
                            asmInstruction.line = firstLine;
                        }
                        else {
                            for (let i = 0; i < generatedSources[asmInstruction.source].length; ++i) {
                                if (asmInstruction.begin >= generatedSources[asmInstruction.source][i].begin &&
                                    asmInstruction.end <= generatedSources[asmInstruction.source][i].end) {
                                    asmInstruction.generatedFunc = generatedSources[asmInstruction.source][i].name;
                                    break;
                                }
                            }
                        }
                    });
                });

                const processOpcodes = (opcodes, indent) => opcodes
                    .map(opcode => {
                        return {
                            text: `${opcode.name.startsWith('tag') ? indent : `\t${indent}`}${opcode.name}${opcode.value !== undefined ? ` ${opcode.value}` : ''} ${opcode.line !== undefined ? opcode.line : ""}:${opcode.source}:${opcode.begin}:${opcode.end}${opcode.generatedFunc !== undefined ? ` ${opcode.generatedFunc}` : ''} `,
                            source: { line: opcode.line, file: null }
                        };
                    });

                return [
                    { text: `// ${_.last(name.split(':'))}` },
                    Object.entries(data.asm['.data']).map(([id, { '.code': code }]) => [
                        processOpcodes(code, ''),
                    ]),
                    { text: '\n' },
                ];
            })
                .flat(Infinity)
                .slice(0, -1),
        };

        /*return {
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
                        Object.entries(data.asm['.data']).map(([id, { '.code': code }]) => [
                            `\t${id}:`,
                            '\t\t.code',
                            processOpcodes(code, '\t\t'),
                        ]),
                        '\n',
                    ];
                })
                .flat(Infinity)
                .slice(0, -1)
                .map(line => ({ text: line })),
        };*/
    }
}
