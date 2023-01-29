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

import Semver from 'semver';

import {BaseCompiler} from '../base-compiler';
import {asSafeVer} from '../utils';

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
            Semver.lt(asSafeVer(this.compiler.semver), '0.8.0', true)
                ? 'asm,ast'
                : 'asm,ast,generated-sources,generated-sources-runtime',
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

        // solc gives us a character range for each asm instruction,
        // so open the input file and figure out what line each
        // character is on
        const inputFile = fs.readFileSync(result.inputFilename);
        let currentLine = 1;
        const charToLine = inputFile.map(c => {
            const line = currentLine;
            if (c === '\n'.codePointAt(0)) {
                ++currentLine;
            }
            return line;
        });
        const hasOldJSONLayout = Semver.lt(asSafeVer(this.compiler.semver), '0.8.0', true);
        const hasGeneratedSources = Semver.gte(asSafeVer(this.compiler.semver), '0.8.0', true);

        const asm = JSON.parse(result.asm);
        return {
            asm: Object.entries(asm.contracts)
                .sort(([_name1, data1], [_name2, data2]) => data1.asm['.code'][0].begin - data2.asm['.code'][0].begin)
                .map(([name, data]) => {
                    // name is in the format of file:contract
                    // e.g. MyFile.sol:MyContract
                    const [sourceName, contractName] = name.split(':');

                    // to make the asm more readable, we rename the
                    // tags (jumpdests) to show what function they're
                    // part of. here we parse the AST so we know what
                    // range of characters belongs to each function.
                    let contractFunctions;
                    // the layout of this JSON has changed between versions...
                    if (hasOldJSONLayout) {
                        contractFunctions = asm.sources[sourceName].AST.children
                            .find(node => {
                                return node.name === 'ContractDefinition' && node.attributes.name === contractName;
                            })
                            .children.filter(node => {
                                return node.name === 'FunctionDefinition';
                            })
                            .map(node => {
                                const [begin, length] = node.src.split(':').map(x => parseInt(x));

                                let name = node.attributes.isConstructor ? 'constructor' : node.attributes.name;

                                // encode the args into the name so we can
                                // differentiate between overloads
                                if (node.children[0].children.length > 0) {
                                    name +=
                                        '_' +
                                        node.children[0].children
                                            .map(paramNode => {
                                                return paramNode.attributes.type;
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
                    } else {
                        contractFunctions = asm.sources[sourceName].AST.nodes
                            .find(node => {
                                return node.nodeType === 'ContractDefinition' && node.name === contractName;
                            })
                            .nodes.filter(node => {
                                return node.nodeType === 'FunctionDefinition';
                            })
                            .map(node => {
                                const [begin, length] = node.src.split(':').map(x => parseInt(x));

                                let name = node.kind === 'constructor' ? 'constructor' : node.name;

                                // encode the args into the name so we can
                                // differentiate between overloads
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
                    }

                    // solc generates some code, for things like detecting
                    // and reverting if a multiplication results in
                    // integer overflow, etc.
                    const processGeneratedSources = generatedSourcesData => {
                        const generatedSources = {};
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

                    // (0.8.x onwards only!)
                    // there are two sets of generated sources, one for the code which deploys
                    // the contract (i.e. the constructor) 'generated-sources', and the code
                    // which is deployed and stored on-chain 'generated-sources-runtime'

                    const generatedSources = hasGeneratedSources
                        ? processGeneratedSources(data['generated-sources'])
                        : {};
                    const generatedSourcesRuntime = hasGeneratedSources
                        ? processGeneratedSources(data['generated-sources-runtime'])
                        : {};

                    const processOpcodes = (opcodes, indent, generatedSources) => {
                        // first iterate the opcodes to find all the tags,
                        // and assign human-readable names to as many of
                        // them as we can
                        const tagNames = {};
                        const processPossibleTagOpcode = (opcode, funcList) => {
                            if (opcode.name === 'tag') {
                                const func = funcList.find(func => {
                                    return opcode.begin >= func.begin && opcode.end <= func.end;
                                });

                                if (func !== undefined) {
                                    // a function can have multiple tags, so append
                                    // a number to each
                                    const tagName = `${func.name}_${func.tagCount}`;

                                    ++func.tagCount;

                                    tagNames[opcode.value] = tagName;
                                    opcode.value = tagName;
                                }
                            }
                        };

                        for (const opcode of opcodes) {
                            // source 0 is the .sol file the user is
                            // editing, everything else is generated
                            // sources (from version 0.8.x onwards).
                            // if source is undefined, then this is
                            // a compiler version which doesn't
                            // provide one (< 0.6.x), but we can
                            // infer 0 in this case.
                            if (opcode.source === 0 || opcode.source === undefined) {
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
