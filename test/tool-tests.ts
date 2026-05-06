// Copyright (c) 2019, Compiler Explorer Authors
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

import {describe, expect, it} from 'vitest';

import {
    getToolchainFlagFromOptions,
    getToolchainPathWithOptionsArr,
    hasToolchainArg,
    removeToolchainArg,
    replaceToolchainArg,
} from '../lib/toolchain-utils.js';
import {ToolEnv} from '../lib/tooling/base-tool.interface.js';
import {BrontoRefactorTool} from '../lib/tooling/bronto-refactor-tool.js';
import {CompilerDropinTool} from '../lib/tooling/compiler-dropin-tool.js';
import {CompilationInfo} from '../types/compilation/compilation.interfaces.js';
import {ToolInfo} from '../types/tool.interfaces.js';

describe('CompilerDropInTool', () => {
    it('Should support llvm based compilers', () => {
        const tool = new CompilerDropinTool({} as ToolInfo, {} as ToolEnv);

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/clang-8.0.0/bin/clang++',
                options: '--gcc-toolchain=/opt/compiler-explorer/gcc-7.2.0',
            },
            options: [],
        } as unknown as CompilationInfo;
        const includeflags: string[] = [];
        const args: string[] = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, [], args, sourcefile);
        expect(orderedArgs).toEqual([
            '--gcc-toolchain=/opt/compiler-explorer/gcc-7.2.0',
            '--gcc-toolchain=/opt/compiler-explorer/gcc-7.2.0',
        ]);
    });

    it('Should support gcc based compilers', () => {
        const tool = new CompilerDropinTool({} as ToolInfo, {} as ToolEnv);

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/gcc-8.0/bin/g++',
                options: '',
            },
            options: [],
        } as unknown as CompilationInfo;
        const includeflags: string[] = [];
        const args: string[] = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, [], args, sourcefile);
        expect(orderedArgs).toEqual([
            '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/gcc-8.0'),
            '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/gcc-8.0'),
        ]);
    });

    it('Should maybe support riscv gcc compilers', () => {
        const tool = new CompilerDropinTool({} as ToolInfo, {} as ToolEnv);

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/riscv64/gcc-8.2.0/riscv64-unknown-linux-gnu/bin/riscv64-unknown-linux-gnu-g++',
                options: '',
            },
            options: [],
        } as unknown as CompilationInfo;
        const includeflags: string[] = [];
        const args: string[] = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, [], args, sourcefile);
        // note: toolchain twice because reasons, see CompilerDropinTool getOrderedArguments()
        expect(orderedArgs).toEqual([
            '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/riscv64/gcc-8.2.0/riscv64-unknown-linux-gnu'),
            '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/riscv64/gcc-8.2.0/riscv64-unknown-linux-gnu'),
        ]);
    });

    it('Should support ICC compilers', () => {
        const tool = new CompilerDropinTool({} as ToolInfo, {} as ToolEnv);

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/intel-2019.1/bin/icc',
                options: '--gxx-name=/opt/compiler-explorer/gcc-8.2.0/bin/g++',
            },
            options: [],
        } as unknown as CompilationInfo;
        const includeflags: string[] = [];
        const args: string[] = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, [], args, sourcefile);
        expect(orderedArgs).toEqual([
            '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/gcc-8.2.0'),
            '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/gcc-8.2.0'),
        ]);
    });

    it('Should not support WINE MSVC compilers', () => {
        const tool = new CompilerDropinTool({} as ToolInfo, {} as ToolEnv);

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/windows/19.14.26423/bin/cl.exe',
                options:
                    '/I/opt/compiler-explorer/windows/10.0.10240.0/ucrt/ ' +
                    '/I/opt/compiler-explorer/windows/19.14.26423/include/',
                internalIncludePaths: ['/opt/compiler-explorer/windows/19.14.26423/include'],
            },
            options: [],
        } as unknown as CompilationInfo;
        const includeflags: string[] = [];
        const args = ['/MD', '/STD:c++latest', '/Ox'];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, [], args, sourcefile);
        expect(orderedArgs).toEqual(false);
    });

    it('Should not support using libc++', () => {
        const tool = new CompilerDropinTool({} as ToolInfo, {} as ToolEnv);

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/clang-concepts-trunk/bin/clang++',
                options: '-stdlib=libc++',
                internalIncludePaths: ['/opt/compiler-explorer/clang-concepts-trunk/something/etc/include'],
            },
            options: [],
        } as unknown as CompilationInfo;
        const includeflags: string[] = [];
        const args: string[] = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, [], args, sourcefile);
        expect(orderedArgs).toEqual(false);
    });

    it('Should support library options', () => {
        const tool = new CompilerDropinTool({} as ToolInfo, {} as ToolEnv);

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/clang-concepts-trunk/bin/clang++',
                options: '--gcc-toolchain=/opt/compiler-explorer/gcc-8.2.0',
                internalIncludePaths: ['/opt/compiler-explorer/clang-concepts-trunk/something/etc/include'],
            },
            options: [],
        } as unknown as CompilationInfo;
        const includeflags: string[] = [];
        const args: string[] = [];
        const sourcefile = 'example.cpp';
        const libOptions = ['-DMYLIBDEF', '-pthread'];

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, libOptions, args, sourcefile);
        expect(orderedArgs).toEqual([
            '--gcc-toolchain=/opt/compiler-explorer/gcc-8.2.0',
            '--gcc-toolchain=/opt/compiler-explorer/gcc-8.2.0',
            '-DMYLIBDEF',
            '-pthread',
        ]);
    });

    it('More toolchain magic', () => {
        const options = [
            '-g',
            '-o',
            'output.s',
            '-mllvm',
            '--x86-asm-syntax=intel',
            '-S',
            '--gcc-toolchain=/opt/compiler-explorer/gcc-12.2.0',
            '-fcolor-diagnostics',
            '-fno-crash-diagnostics',
            '/app/example.cpp',
        ];

        expect(hasToolchainArg(options)).toBe(true);

        expect(getToolchainFlagFromOptions(options)).toEqual('--gcc-toolchain=');

        const newOptions = removeToolchainArg(options);
        expect(hasToolchainArg(newOptions)).toBe(false);
    });

    it('Should be able to swap toolchain', () => {
        const exe = '/opt/compiler-explorer/clang-16.0.0/bin/clang++';
        const options = [
            '-g',
            '-o',
            'output.s',
            '-mllvm',
            '--x86-asm-syntax=intel',
            '-S',
            '--gcc-toolchain=/opt/compiler-explorer/gcc-12.2.0',
            '-fcolor-diagnostics',
            '-fno-crash-diagnostics',
            '/app/example.cpp',
        ];

        const toolchain = getToolchainPathWithOptionsArr(exe, options);
        expect(toolchain).toEqual('/opt/compiler-explorer/gcc-12.2.0');

        const replacedOptions = replaceToolchainArg(options, '/opt/compiler-explorer/gcc-11.1.0');
        expect(replacedOptions).toEqual([
            '-g',
            '-o',
            'output.s',
            '-mllvm',
            '--x86-asm-syntax=intel',
            '-S',
            '--gcc-toolchain=' + path.normalize('/opt/compiler-explorer/gcc-11.1.0'),
            '-fcolor-diagnostics',
            '-fno-crash-diagnostics',
            '/app/example.cpp',
        ]);
    });
});

describe('BrontoRefactorTool', () => {
    function runTool(stdout: string, stderr: string) {
        const tool = new BrontoRefactorTool({} as ToolInfo, {} as ToolEnv);
        return tool.convertResult(
            {
                code: 0,
                okToCache: true,
                timedOut: false,
                truncated: false,
                stdout,
                stderr,
                filenameTransform: f => f,
                execTime: 0,
            },
            'example.cpp',
        );
    }

    // Output is refactored source code, not diagnostics. The default FileWithLine matcher
    // would tag e.g. `r.fetch_add(2, ...)` as a `filename:line:` reference and surface it
    // as an error in the editor.
    it('Should not tag refactored source as diagnostics', () => {
        const refactoredSource = [
            '#include <atomic>',
            'void foo(std::atomic<int>& r) {',
            '  r.fetch_add(2, std::memory_order_relaxed);',
            '  r.fetch_add(4, std::memory_order_relaxed);',
            '}',
        ].join('\n');

        const result = runTool(refactoredSource, '');

        const tagged = result.stdout.filter(line => line.tag);
        expect(tagged).toEqual([]);
    });

    // Make sure dropping FileWithLineMessage didn't blunt the parser too much: real
    // `<source>:N:M:` diagnostics on stderr should still be tagged via SourceWithLineMessage.
    it('Should still tag real diagnostics from stderr', () => {
        const result = runTool('', 'example.cpp:10:5: error: something went wrong\n');

        expect(result.stderr).toHaveLength(1);
        expect(result.stderr[0].tag).toEqual({
            file: 'example.cpp',
            line: 10,
            column: 5,
            text: 'error: something went wrong',
            severity: 3,
        });
    });
});
