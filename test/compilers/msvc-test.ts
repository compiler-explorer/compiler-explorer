// Copyright (c) 2024, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import {ClangCLCompiler, Win32VcCompiler} from '../../lib/compilers/index.js';
import {makeCompilationEnvironment} from '../utils.js';

describe('msvc tests', () => {
    const languages = {'c++': {id: 'c++'}};

    const info = {
        exe: 'foobar',
        remote: true,
        lang: 'c++',
        ldPath: [],
        options: '/EHsc /utf-8',
    };

    describe('utf8 replacement', async () => {
        const msvc = new Win32VcCompiler(info as any, makeCompilationEnvironment({languages}));
        it('should do the normal situation', async () => {
            const prepared: string[] = msvc.prepareArguments([], {}, {}, 'example.cpp', 'output.s', [], []);
            expect(prepared).toEqual([
                '/nologo',
                '/FA',
                '/c',
                '/Faoutput.s',
                '/Fooutput.s.obj',
                '/Z7',
                '/EHsc',
                '/utf-8',
                'example.cpp',
            ]);
        });

        it('should replace arguments', async () => {
            const prepared: string[] = msvc.prepareArguments(
                ['/O2', '/source-charset:windows-1252'],
                {},
                {},
                'example.cpp',
                'output.s',
                [],
                [],
            );
            expect(prepared).toEqual([
                '/nologo',
                '/FA',
                '/c',
                '/Faoutput.s',
                '/Fooutput.s.obj',
                '/Z7',
                '/EHsc',
                '/O2',
                '/source-charset:windows-1252',
                'example.cpp',
            ]);
        });
    });

    describe('debug information', async () => {
        const msvc = new Win32VcCompiler(info as any, makeCompilationEnvironment({languages}));
        it('should drop debug options that need mspdbsrv', async () => {
            const prepared: string[] = msvc.prepareArguments(
                ['/Zi', '-ZI', '/FS', '/Zc:__cplusplus', '/Z7'],
                {},
                {},
                'example.cpp',
                'output.s',
                [],
                [],
            );
            expect(prepared).toEqual([
                '/nologo',
                '/FA',
                '/c',
                '/Faoutput.s',
                '/Fooutput.s.obj',
                '/Z7',
                '/EHsc',
                '/utf-8',
                '/Zc:__cplusplus',
                '/Z7',
                'example.cpp',
            ]);
        });

        it('should use /Z7 when building a binary', async () => {
            const options = msvc.optionsForFilter({binary: true} as any, 'output.s');
            expect(options).toContain('/Z7');
            expect(options).not.toContain('/Zi');
        });

        it('should have the linker write the map file', async () => {
            const prepared = msvc.prepareArguments(['/O2'], {binary: true}, {}, 'example.cpp', 'output.s', [], []);
            expect(prepared.slice(prepared.indexOf('example.cpp'))).toEqual([
                'example.cpp',
                '/link',
                '/MAP:output.s.map',
            ]);
            expect(prepared.some(option => option.startsWith('/Fm'))).toBe(false);
        });

        it('should keep the map file when the user passes their own linker options', async () => {
            const prepared = msvc.prepareArguments(
                ['/O2', '/link', '/STACK:4096'],
                {binary: true},
                {},
                'example.cpp',
                'output.s',
                [],
                [],
            );
            expect(prepared.slice(prepared.indexOf('example.cpp'))).toEqual([
                'example.cpp',
                '/link',
                '/MAP:output.s.map',
                '/STACK:4096',
            ]);
        });

        it('should not ask for a map file from a compiler that does not support one', async () => {
            const compiler = new Win32VcCompiler(info as any, makeCompilationEnvironment({languages}));
            compiler.supportsMapFile = false;
            const filters: any = {binary: true};
            const prepared = compiler.prepareArguments([], filters, {}, 'example.cpp', 'output.s', [], []);
            expect(prepared.slice(prepared.indexOf('example.cpp'))).toEqual(['example.cpp', '/link']);
            expect(filters.preProcessBinaryAsmLines).toBeUndefined();
        });

        it('should embed debug information in CMake builds', async () => {
            const args = msvc.getExtraCMakeArgs({});
            expect(args).toContain('-DCMAKE_POLICY_DEFAULT_CMP0141=NEW');
            expect(args).toContain('-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded');
            expect(args).toContain('-DCMAKE_POLICY_DEFAULT_CMP0091=NEW');
            expect(args).toContain('-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL');
            expect(args).toContain('-DCMAKE_EXE_LINKER_FLAGS_INIT=/MAP');
        });

        it('should point CMake at clang-cl and the LLVM tools next to it', async () => {
            const clangCl = new ClangCLCompiler(
                {
                    ...info,
                    exe: 'Z:/compilers/clang-cl-22.1.6/bin/clang-cl.exe',
                    options: '/winsdkdir Z:/compilers/windows-kits-10 /EHsc /vctoolsdir Z:/compilers/msvc/14.50',
                } as any,
                makeCompilationEnvironment({languages}),
            );
            const args = clangCl.getExtraCMakeArgs({} as any);
            expect(args).toContain('-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded');
            expect(args).toContain('-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL');
            expect(args).toContain('-DCMAKE_CXX_COMPILER=Z:/compilers/clang-cl-22.1.6/bin/clang-cl.exe');
            expect(args).toContain('-DCMAKE_LINKER=Z:/compilers/clang-cl-22.1.6/bin/lld-link.exe');
            expect(args).toContain('-DCMAKE_RC_COMPILER=Z:/compilers/clang-cl-22.1.6/bin/llvm-rc.exe');
            expect(args).toContain('-DCMAKE_MT=Z:/compilers/clang-cl-22.1.6/bin/llvm-mt.exe');
            expect(args).toContain(
                '-DCMAKE_EXE_LINKER_FLAGS_INIT=/MAP /vctoolsdir:Z:/compilers/msvc/14.50 /winsdkdir:Z:/compilers/windows-kits-10',
            );
        });

        it('should not ask CMake builds for a map file from a compiler that does not support one', async () => {
            const compiler = new Win32VcCompiler(info as any, makeCompilationEnvironment({languages}));
            compiler.supportsMapFile = false;
            expect(compiler.getExtraCMakeArgs({})).not.toContain('-DCMAKE_EXE_LINKER_FLAGS_INIT=/MAP');
        });
    });

    describe('binary mode', () => {
        // Real objdump output of an executable linking a static library, with its linker map
        const staticLibFixture: {name: string; encoding: BufferEncoding} = {
            name: 'test/maps/squareprintfstaticlib',
            encoding: 'utf16le',
        };
        // An incrementally linked /Z7 build as CE makes them today: .text starts with int3 padding, then jump thunks
        const incrementalFixture = {name: 'test/maps/incremental-z7', encoding: 'utf8' as BufferEncoding};
        // clang-cl 22 linked by lld-link, its map asked for with /MAP
        const clangClFixture = {name: 'test/maps/clang-cl', encoding: 'utf8' as BufferEncoding};

        async function disassemble(
            objdumperType: string,
            libraryCode: boolean,
            fixture = staticLibFixture,
            Compiler: typeof Win32VcCompiler | typeof ClangCLCompiler = Win32VcCompiler,
        ) {
            const msvc = new Compiler(
                {...info, objdumperType} as any,
                makeCompilationEnvironment({
                    languages,
                    props: {binaryHideFuncRe: '^(__.*|_(init|start|fini))$'},
                }),
            );
            const filters: any = {binary: true, labels: true, libraryCode, directives: true, commentOnly: true};
            msvc.optionsForFilter(filters, fixture.name);
            const asm = await fs.readFile(`${fixture.name}.asm`, fixture.encoding);
            const result = await msvc.processAsm({asm}, filters);
            return result.asm.map((line: {text: string}) => line.text);
        }

        const functionLabels = (lines: string[]) => lines.filter(line => /^[^\s$L].*:$/.test(line));

        it("keeps only the program's own functions when filtering library code, named from the map file", async () => {
            const lines = await disassemble('', true);
            expect(functionLabels(lines)).toEqual(['_vfprintf_l:', 'printf:', 'main:', '?square@@YAHH@Z:']);
        });

        it('names the library code too when it is shown', async () => {
            const labels = functionLabels(await disassemble('', false));
            expect(labels).toContain('main:');
            expect(labels).toContain('mainCRTStartup:');
            expect(labels).toContain('?getValue@String@Groundfloor@@QEBAPEADXZ:');
        });

        it('filters an incrementally linked build whose .text starts with padding', async () => {
            const lines = await disassemble('', true, incrementalFixture);
            expect(functionLabels(lines)).toEqual(['?square@@YAHH@Z:', '_vfprintf_l:', 'main:', 'printf:']);
        });

        it('filters clang-cl output with the map lld-link writes', async () => {
            const lines = await disassemble('', true, clangClFixture, ClangCLCompiler);
            expect(functionLabels(lines)).toEqual(['?square@@YAHH@Z:', 'main:', 'printf:']);
        });

        describe('of a CMake build', () => {
            // A CMake build whose executable links the static library target sq, and the map /MAP wrote next to it
            const cmakeFixture = 'test/maps/cmake-msvc';
            let buildPath: string;

            beforeEach(async () => {
                buildPath = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-msvc-cmake-'));
                await fs.writeFile(path.join(buildPath, 'the_executable.exe'), 'MZ');
            });

            afterEach(async () => {
                await fs.rm(buildPath, {recursive: true, force: true});
            });

            async function disassembleArtifact() {
                const msvc = new Win32VcCompiler(
                    info as any,
                    makeCompilationEnvironment({languages, props: {binaryHideFuncRe: '^(__.*|_(init|start|fini))$'}}),
                );
                const filters: any = {binary: true, labels: true, libraryCode: true, directives: true};
                await msvc.prepareArtifactDisassembly(path.join(buildPath, 'the_executable.exe'), filters);
                const asm = await fs.readFile(`${cmakeFixture}.asm`, 'utf8');
                const result = await msvc.processAsm({asm}, filters);
                return {filters, labels: functionLabels(result.asm.map((line: {text: string}) => line.text))};
            }

            async function writeMap() {
                await fs.copyFile(`${cmakeFixture}.map`, path.join(buildPath, 'the_executable.map'));
            }

            it("keeps the project's own code, from its library targets too", async () => {
                await writeMap();
                await fs.mkdir(path.join(buildPath, 'lib'));
                await fs.writeFile(path.join(buildPath, 'lib', 'sq.lib'), '!<arch>');
                const {labels} = await disassembleArtifact();
                expect(labels).toEqual(['_vfprintf_l:', 'main:', 'printf:', '?square@@YAHH@Z:']);
            });

            it('counts a library it did not build as library code', async () => {
                await writeMap();
                const {labels} = await disassembleArtifact();
                expect(labels).toEqual(['_vfprintf_l:', 'main:', 'printf:']);
            });

            it('leaves the disassembly alone without a map', async () => {
                const {filters} = await disassembleArtifact();
                expect(filters.preProcessBinaryAsmLines).toBeUndefined();
            });
        });

        it('passes dumpbin output through as it is', async () => {
            const lines = await disassemble('vc', true);
            expect(functionLabels(lines)).toEqual([]);
            expect(lines.length).toBeGreaterThan(5000);
        });
    });

    describe('clang-cl option discovery', () => {
        it('should read the options from -help, as clang-cl rejects --help', async () => {
            const clangCl = new ClangCLCompiler(
                {...info, exe: 'Z:/compilers/clang-cl-22.1.6/bin/clang-cl.exe'} as any,
                makeCompilationEnvironment({languages}),
            ) as any;
            clangCl.execCompilerCached = async (exe: string, args: string[]) =>
                args[0] === '-help'
                    ? {
                          code: 0,
                          stdout: '  -fcolor-diagnostics\n  -fno-crash-diagnostics\n  --target=<value>\n',
                          stderr: '',
                      }
                    : {code: 1, stdout: '', stderr: 'clang-cl: error: no input files'};
            const result = await clangCl.argParser.parse();
            expect(result.compiler.options).toContain('-fcolor-diagnostics');
            expect(result.compiler.options).toContain('-fno-crash-diagnostics');
            expect(result.compiler.supportsTargetIs).toBe(true);
        });
    });
});
