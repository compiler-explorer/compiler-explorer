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

import * as fs from 'fs';
import path from 'path';

import {BaseCompiler} from '../base-compiler';

import {ClangParser} from './argument-parsers';

export class SolidityCompiler extends BaseCompiler {
    static get key() {
        return 'solidity';
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    getArgumentParser() {
        return ClangParser;
    }

    optionsForFilter(filters, outputFilename, userOptions) {
        return [
            // We use --combined-json instead of `--asm-json` to have compacted json
            '--combined-json',
            'asm,ast,generated-sources,generated-sources-runtime',
            '-o',
            'contracts',
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

        const inputFile = fs.readFileSync(result.inputFilename);
        let currentLine = 1;
        const charToLine = inputFile.map(c => {
            const line = currentLine;
            if (c === '\n'.codePointAt(0)) {
                ++currentLine;
            }
            return line;
        });

        const asm = JSON.parse(result.asm);
        return {
            asm: Object.entries(asm.contracts)
                .sort(([_name1, data1], [_name2, data2]) => data1.asm['.code'][0].begin - data2.asm['.code'][0].begin)
                .map(([name, data]) => {
                    const [sourceName, contractName] = name.split(':');

                    const contractFunctions = asm.sources[sourceName].AST.nodes
                        .find(node => {
                            return node.nodeType === 'ContractDefinition' && node.name === contractName;
                        })
                        .nodes.filter(node => {
                            return node.nodeType === 'FunctionDefinition';
                        })
                        .map(node => {
                            const [begin, length] = node.src.split(':').map(x => parseInt(x));

                            // encode the args into the name so we can
                            // differentiate between overloads
                            let name = node.kind === 'constructor' ? 'constructor' : node.name;

                            if (node.parameters.parameters.length > 0) {
                                name +=
                                    '_' +
                                    node.parameters.parameters
                                        .map(paramNode => {
                                            return paramNode.typeName.name;
                                        })
                                        .join('_');
                            }

                            return {
                                name: name,
                                begin: begin,
                                end: begin + length,
                                tagCount: 0,
                            };
                        });

                    const processGeneratedSources = generatedSourcesData => {
                        let generatedSources = {};
                        for (const generatedSource of generatedSourcesData) {
                            generatedSources[generatedSource.id] = generatedSource.ast.statements.map(statement => {
                                const [begin, length] = statement.src.split(':').map(x => parseInt(x));
                                return {
                                    name: statement.name,
                                    begin: begin,
                                    end: begin + length,
                                    tagCount: 0,
                                };
                            });
                        }
                        return generatedSources;
                    };
                    const generatedSources = processGeneratedSources(data['generated-sources']);
                    const generatedSourcesRuntime = processGeneratedSources(data['generated-sources-runtime']);

                    const processOpcodes = (opcodes, indent, generatedSources) => {
                        let tagNames = {};
                        const processPossibleTagOpcode = (opcode, funcList) => {
                            if (opcode.name === 'tag') {
                                const func = funcList.find(func => {
                                    return opcode.begin >= func.begin && opcode.end <= func.end;
                                });

                                if (func !== undefined) {
                                    const tagName = `${func.name}_${func.tagCount}`;

                                    ++func.tagCount;

                                    tagNames[opcode.value] = tagName;
                                    opcode.value = tagName;
                                }
                            }
                        };

                        for (const opcode of opcodes) {
                            if (opcode.source === 0) {
                                opcode.line = charToLine[opcode.begin];

                                processPossibleTagOpcode(opcode, contractFunctions);
                            } else {
                                processPossibleTagOpcode(opcode, generatedSources[opcode.source]);
                            }
                        }

                        return opcodes.map(opcode => {
                            const name = `${opcode.name.startsWith('tag') ? indent : `${indent}\t`}${opcode.name}`;

                            let value = opcode.value || '';
                            if (opcode.name === 'PUSH [tag]') {
                                if (tagNames[value] !== undefined) {
                                    value = tagNames[value];
                                }
                            }

                            return {
                                text: `${name} ${value}`,
                                source: {line: opcode.line, file: null},
                            };
                        });
                    };

                    return [
                        {text: `// ${contractName}`},
                        // .code section is the code only run when deploying - the constructor
                        {text: '.code'},
                        processOpcodes(data.asm['.code'], '', generatedSources),
                        {text: ''},
                        // .data section is deployed bytecode - everything else
                        {text: '.data'},
                        Object.entries(data.asm['.data']).map(([id, {'.code': code}]) => [
                            {text: `\t${id}:`},
                            processOpcodes(code, '\t', generatedSourcesRuntime),
                        ]),
                    ];
                })
                .flat(Infinity),
        };
    }
}
