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
            '--combined-json', 'asm,ast,generated-sources-runtime', // We use it instead of `--asm-json` to have compacted json
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
                const [sourceName, contractName] = name.split(':');

                const contractFunctions = asm.sources[sourceName].AST.nodes
                    .find(node => {
                        return node.nodeType == "ContractDefinition" &&
                            node.name == contractName;
                    }).nodes
                    .filter(node => {
                        return node.nodeType == "FunctionDefinition"
                    })
                    .map(node => {
                        const [begin, length] = node.src.split(":").map(x => parseInt(x));

                        // encode the args into the name so we can
                        // differentiate between overloads
                        let name = node.name;
                        if (node.parameters.parameters.length > 0) {
                            name += '_' + node.parameters.parameters.map(paramNode => {
                                return paramNode.typeName.name;
                            }).join('_');
                        }

                        return {
                            name: name,
                            begin: begin,
                            end: begin + length,
                            tagCount: 0
                        }
                    });

                let generatedSources = {};
                data["generated-sources-runtime"].forEach(generatedSource => {
                    generatedSources[generatedSource.id] = generatedSource.ast.statements.map(statement => {
                        const [begin, length] = statement.src.split(":").map(x => parseInt(x));
                        return {
                            name: statement.name,
                            begin: begin,
                            end: begin + length,
                            tagCount: 0
                        };
                    });
                });

                let tagNames = {};
                Object.entries(data.asm['.data']).forEach(([id, { '.code': code }]) => {
                    const processPossibleTagInstruction = function (asmInstruction, funcList) {
                        if (asmInstruction.name == "tag") {
                            const func = funcList.find(func => {
                                return asmInstruction.begin >= func.begin &&
                                    asmInstruction.end <= func.end;
                            });

                            if (func !== undefined) {
                                const tagName = `${func.name}_${func.tagCount}`;

                                ++func.tagCount;

                                tagNames[asmInstruction.value] = tagName;
                                asmInstruction.value = tagName;
                            }
                        }
                    };

                    code.forEach((asmInstruction) => {
                        if (asmInstruction.source == 0) {
                            asmInstruction.line = charToLine[asmInstruction.begin];

                            processPossibleTagInstruction(asmInstruction, contractFunctions);
                        }
                        else {
                            processPossibleTagInstruction(asmInstruction, generatedSources[asmInstruction.source]);
                        }
                    });
                });

                const processOpcodes = (opcodes, indent) => opcodes
                    .map(opcode => {
                        let value = opcode.value;
                        if (opcode.name == "PUSH [tag]") {
                            if (tagNames[value] !== undefined) {
                                value = tagNames[value];
                            }
                        }
                        return {
                            text: `${opcode.name.startsWith('tag') ? indent : `\t${indent}`}${opcode.name}${value !== undefined ? ` ${value}` : ''}`,
                            source: { line: opcode.line, file: null }
                        };
                    });

                return [
                    { text: `// ${contractName}` },
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
