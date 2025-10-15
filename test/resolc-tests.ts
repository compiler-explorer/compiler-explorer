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

import path from 'node:path';

import {beforeAll, describe, expect, it} from 'vitest';

import type {CompilationEnvironment} from '../lib/compilation-env.js';
import {ResolcParser} from '../lib/compilers/argument-parsers.js';
import {ResolcCompiler} from '../lib/compilers/index.js';
import type {ParsedAsmResult, ParsedAsmResultLine} from '../types/asmresult/asmresult.interfaces.js';
import type {CompilerInfo} from '../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, makeFakeLlvmIrBackendOptions} from './utils.js';

const languages = {
    solidity: {id: 'solidity' as LanguageKey},
    yul: {id: 'yul' as LanguageKey},
};

describe('Resolc', () => {
    let env: CompilationEnvironment;
    const expectedSolcExe = '/opt/compiler-explorer/solc-0.8.30/solc';

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    function makeCompiler(compilerInfo: Partial<CompilerInfo>): ResolcCompiler {
        return new ResolcCompiler(makeFakeCompilerInfo(compilerInfo), env);
    }

    function expectCorrectOutputFilenames(
        compiler: ResolcCompiler,
        inputFilename: string,
        expectedFilenameWithoutExtension: string,
    ): void {
        const defaultOutputFilename = `${expectedFilenameWithoutExtension}.pvmasm`;
        expect(compiler.getOutputFilename(path.normalize('test/resolc'))).toEqual(defaultOutputFilename);

        let llvmIrBackendOptions = makeFakeLlvmIrBackendOptions({showOptimized: true});
        expect(compiler.getIrOutputFilename(inputFilename, undefined, llvmIrBackendOptions)).toEqual(
            `${expectedFilenameWithoutExtension}.optimized.ll`,
        );

        llvmIrBackendOptions = makeFakeLlvmIrBackendOptions({showOptimized: false});
        expect(compiler.getIrOutputFilename(inputFilename, undefined, llvmIrBackendOptions)).toEqual(
            `${expectedFilenameWithoutExtension}.unoptimized.ll`,
        );

        expect(compiler.getObjdumpOutputFilename(defaultOutputFilename)).toEqual(
            `${expectedFilenameWithoutExtension}.o`,
        );
    }

    describe('Common', () => {
        it('should return correct key', () => {
            expect(ResolcCompiler.key).toEqual('resolc');
        });

        it('should return Solc executable dependency path', () => {
            expect(ResolcCompiler.solcExe).toEqual(expectedSolcExe);
        });
    });

    describe('From Solidity', () => {
        const compilerInfo = {
            exe: 'resolc',
            lang: languages.solidity.id,
        };

        it('should instantiate successfully', () => {
            const compiler = makeCompiler(compilerInfo);
            expect(compiler.lang.id).toEqual(compilerInfo.lang);
        });

        it('should use Resolc argument parser', () => {
            const compiler = makeCompiler(compilerInfo);
            expect(compiler.getArgumentParserClass()).toBe(ResolcParser);
        });

        it('should use debug options', () => {
            const compiler = makeCompiler(compilerInfo);
            expect(compiler.optionsForFilter({})).toEqual([
                '-g',
                '--solc',
                expectedSolcExe,
                '--overwrite',
                '--debug-output-dir',
                'artifacts',
            ]);
        });

        it('should generate output filenames', () => {
            const compiler = makeCompiler(compilerInfo);
            const filenameWithoutExtension = path.normalize('test/resolc/artifacts/test_resolc_example.sol.Square');
            const inputFilename = path.normalize('test/resolc/example.sol');
            expectCorrectOutputFilenames(compiler, inputFilename, filenameWithoutExtension);
        });

        describe('To RISC-V', () => {
            const filters: Partial<ParseFiltersAndOutputOptions> = {
                binaryObject: true,
                libraryCode: true,
            };

            function getExpectedParsedOutputHeader(): ParsedAsmResultLine[] {
                const header =
                    '; RISC-V (64 bits) Assembly:\n' +
                    '; --------------------------\n' +
                    '; To see the PolkaVM assembly instead,\n' +
                    '; enable "Compile to binary object".\n' +
                    '; --------------------------';

                return header.split('\n').map(line => ({text: line}));
            }

            it('should remove orphaned labels', async () => {
                const compiler = makeCompiler(compilerInfo);

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
                        __last: 8,
                    },
                };

                const expected: ParsedAsmResult = {
                    asm: [
                        ...getExpectedParsedOutputHeader(),
                        {
                            text: '__entry:',
                        },
                        {
                            text: '        addi	sp, sp, -0x10',
                        },
                        {
                            text: '        sd	ra, 0x8(sp)',
                        },
                    ],
                };

                const result = await compiler.postProcessAsm(parsedAsm, filters);
                expect(result.asm.length).toEqual(expected.asm.length);
                expect(result.asm).toMatchObject(expected.asm);
            });

            it('should remove invalid Solidity <--> RISC-V source mappings', async () => {
                const compiler = makeCompiler(compilerInfo);

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
                        ...getExpectedParsedOutputHeader(),
                        {
                            text: '.Lpcrel_hi4:',
                            source: null,
                        },
                        {
                            text: '        auipc	a1, 0x0',
                            source: null,
                        },
                        {
                            text: '        addi	sp, sp, -0x60',
                            source: null,
                        },
                        {
                            text: '        sd	ra, 0x58(sp)',
                            source: null,
                        },
                        {
                            text: '        jalr	ra <.Lpcrel_hi4+0x3a>',
                            source: null,
                        },
                    ],
                };

                const result = await compiler.postProcessAsm(parsedAsm, filters);
                expect(result.asm.length).toEqual(expected.asm.length);
                expect(result.asm).toMatchObject(expected.asm);
            });
        });
    });

    describe('From Yul', () => {
        const compilerInfo = {
            exe: 'resolc',
            lang: languages.yul.id,
        };

        it('should instantiate successfully', () => {
            const compiler = makeCompiler(compilerInfo);
            expect(compiler.lang.id).toEqual(compilerInfo.lang);
        });

        it('should use debug options', () => {
            const compiler = makeCompiler(compilerInfo);
            expect(compiler.optionsForFilter({})).toEqual([
                '-g',
                '--solc',
                expectedSolcExe,
                '--overwrite',
                '--debug-output-dir',
                'artifacts',
                '--yul',
            ]);
        });

        it('should generate output filenames', () => {
            const compiler = makeCompiler(compilerInfo);
            const filenameWithoutExtension = path.normalize('test/resolc/artifacts/test_resolc_example.yul.Square');
            const inputFilename = path.normalize('test/resolc/example.yul');
            expectCorrectOutputFilenames(compiler, inputFilename, filenameWithoutExtension);
        });
    });
});
