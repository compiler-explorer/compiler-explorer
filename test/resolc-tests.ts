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

import {beforeAll, describe, expect, it} from 'vitest';

import type {CompilationEnvironment} from '../lib/compilation-env.js';
import {ResolcParser} from '../lib/compilers/argument-parsers.js';
import {ResolcCompiler} from '../lib/compilers/index.js';
import type {ParsedAsmResult} from '../types/asmresult/asmresult.interfaces.js';
import type {CompilerInfo} from '../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    solidity: {id: 'solidity'},
    yul: {id: 'yul'},
};

const solidityInfo = {
    exe: 'resolc',
    lang: languages.solidity.id as LanguageKey,
    name: 'resolc 0.4.0 (RISC-V 64-bits)',
};

const yulInfo = {
    exe: 'resolc',
    lang: languages.yul.id as LanguageKey,
    name: 'resolc 0.4.0 (RISC-V 64-bits)',
};

describe('Resolc', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    const makeCompiler = (compilerInfo: Partial<CompilerInfo>) =>
        new ResolcCompiler(makeFakeCompilerInfo(compilerInfo), env);

    describe('Common', () => {
        it('should return correct key', () => {
            expect(ResolcCompiler.key).toEqual('resolc');
        });
    });

    describe('Solidity', () => {
        it('should instantiate successfully', () => {
            const compiler = makeCompiler(solidityInfo);
            expect(compiler.lang.id).toEqual(solidityInfo.lang);
        });

        it('should use Resolc argument parser', () => {
            const compiler = makeCompiler(solidityInfo);
            expect(compiler.getArgumentParserClass()).toBe(ResolcParser);
        });

        it('should use debug options', () => {
            const compiler = makeCompiler(solidityInfo);
            expect(compiler.optionsForFilter({})).toEqual(['-g', '--overwrite', '--debug-output-dir', 'artifacts']);
        });

        it('should generate output filenames', () => {
            const compiler = makeCompiler(solidityInfo);
            const defaultOutputFilename = 'test/resolc/artifacts/test_resolc_example.sol.Square.pvmasm';
            expect(compiler.getOutputFilename('test/resolc')).toEqual(defaultOutputFilename);
            expect(compiler.getIrOutputFilename('test/resolc/example.sol')).toEqual(
                'test/resolc/artifacts/test_resolc_example.sol.Square.unoptimized.ll',
            );
            expect(compiler.getObjdumpOutputFilename(defaultOutputFilename)).toEqual(
                'test/resolc/artifacts/test_resolc_example.sol.Square.o',
            );
        });

        it('should remove orphaned labels', async () => {
            const compiler = makeCompiler(solidityInfo);

            const filters: Partial<ParseFiltersAndOutputOptions> = {
                binaryObject: true,
                libraryCode: true,
            };

            const parsedAsm: ParsedAsmResult = {
                asm: [
                    {
                        // Orphan
                        text: 'memmove:',
                    },
                    {
                        // Orphan
                        text: '.LBB34_5:',
                    },
                    {
                        // Orphan
                        text: 'memset:',
                    },
                    {
                        // Orphan
                        text: '.LBB35_2:',
                    },
                    {
                        text: '__entry:',
                    },
                    {
                        text: ' addi	sp, sp, -0x10',
                    },
                    {
                        text: ' sd	ra, 0x8(sp)',
                    },
                    {
                        // Orphan
                        text: '__last:',
                    },
                ],
                labelDefinitions: {
                    memmove: 1,
                    ['.LBB34_5']: 2,
                    memset: 3,
                    ['.LBB35_2']: 4,
                    __entry: 5,
                    __last: 6,
                },
            };

            const expected: ParsedAsmResult = {
                asm: [
                    {
                        text: '__entry:',
                    },
                    {
                        text: '\t addi	sp, sp, -0x10',
                    },
                    {
                        text: '\t sd	ra, 0x8(sp)',
                    },
                ],
            };

            const result = await compiler.postProcessAsm(parsedAsm, filters);
            expect(result.asm.length).toEqual(expected.asm.length);
            expect(result.asm).toMatchObject(expected.asm);
        });

        it('should remove Solidity <--> RISC-V source mappings', async () => {
            const compiler = makeCompiler(solidityInfo);

            const filters: Partial<ParseFiltersAndOutputOptions> = {
                binaryObject: true,
                libraryCode: true,
            };

            const parsedAsm: ParsedAsmResult = {
                asm: [
                    {
                        text: '.Lpcrel_hi4:',
                        source: null,
                    },
                    {
                        text: ' auipc	a1, 0x0',
                        source: null,
                    },
                    {
                        text: ' addi	sp, sp, -0x60',
                        source: {
                            line: 1,
                            file: null,
                        },
                    },
                    {
                        text: ' sd	ra, 0x58(sp)',
                        source: {
                            line: 1,
                            file: null,
                        },
                    },
                    {
                        text: ' jalr	ra <.Lpcrel_hi4+0x3a>',
                        source: {
                            line: 7,
                            file: null,
                        },
                    },
                ],
                labelDefinitions: {['.Lpcrel_hi4']: 1},
            };

            const expected: ParsedAsmResult = {
                asm: [
                    {
                        text: '.Lpcrel_hi4:',
                        source: null,
                    },
                    {
                        text: '\t auipc	a1, 0x0',
                        source: null,
                    },
                    {
                        text: '\t addi	sp, sp, -0x60',
                        source: null,
                    },
                    {
                        text: '\t sd	ra, 0x58(sp)',
                        source: null,
                    },
                    {
                        text: '\t jalr	ra <.Lpcrel_hi4+0x3a>',
                        source: null,
                    },
                ],
            };

            const result = await compiler.postProcessAsm(parsedAsm, filters);
            expect(result.asm.length).toEqual(expected.asm.length);
            expect(result.asm).toMatchObject(expected.asm);
        });
    });

    describe('Yul', () => {
        it('should instantiate successfully', () => {
            const compiler = makeCompiler(yulInfo);
            expect(compiler.lang.id).toEqual(yulInfo.lang);
        });

        it('should use debug options', () => {
            const compiler = makeCompiler(yulInfo);
            expect(compiler.optionsForFilter({})).toEqual([
                '-g',
                '--overwrite',
                '--debug-output-dir',
                'artifacts',
                '--yul',
            ]);
        });

        it('should generate output filenames', () => {
            const compiler = makeCompiler(yulInfo);
            const defaultOutputFilename = 'test/resolc/artifacts/test_resolc_example.yul.Square.pvmasm';
            expect(compiler.getOutputFilename('test/resolc')).toEqual(defaultOutputFilename);
            expect(compiler.getIrOutputFilename('test/resolc/example.sol')).toEqual(
                'test/resolc/artifacts/test_resolc_example.yul.Square.unoptimized.ll',
            );
            expect(compiler.getObjdumpOutputFilename(defaultOutputFilename)).toEqual(
                'test/resolc/artifacts/test_resolc_example.yul.Square.o',
            );
        });
    });
});
