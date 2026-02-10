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

import {BaseCompiler} from '../lib/base-compiler.js';
import {DefaultObjdumper} from '../lib/objdumper/default.js';
import {LlvmObjdumper} from '../lib/objdumper/llvm.js';
import type {CompilationResult, ExecutionOptions} from '../types/compilation/compilation.interfaces.js';
import type {CompilerInfo} from '../types/compiler.interfaces.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

describe('Objdumper', () => {
    describe('BaseObjdumper', () => {
        it('should execute objdump successfully', async () => {
            const objdumper = new DefaultObjdumper();

            // Mock exec function
            const mockExec = async (
                filepath: string,
                args: string[],
                options: ExecutionOptions,
            ): Promise<UnprocessedExecResult> => {
                return {
                    code: 0,
                    okToCache: true,
                    filenameTransform: (f: string) => f,
                    stdout: 'test assembly output',
                    stderr: '',
                    execTime: 100,
                    timedOut: false,
                    truncated: false,
                };
            };

            const result = await objdumper.executeObjdump(
                '/usr/bin/objdump',
                ['-d', 'test.o'],
                {maxOutput: 1024},
                mockExec,
            );

            expect(result.code).toBe(0);
            expect(result.asm).toBe('test assembly output');
            expect(result.objdumpTime).toBe('100');
        });

        it('should handle objdump failure', async () => {
            const objdumper = new DefaultObjdumper();

            // Mock exec function that fails
            const mockExec = async (
                filepath: string,
                args: string[],
                options: ExecutionOptions,
            ): Promise<UnprocessedExecResult> => {
                return {
                    code: 1,
                    okToCache: false,
                    filenameTransform: (f: string) => f,
                    stdout: '',
                    stderr: 'objdump: test.o: No such file',
                    execTime: 50,
                    timedOut: false,
                    truncated: false,
                };
            };

            const result = await objdumper.executeObjdump(
                '/usr/bin/objdump',
                ['-d', 'test.o'],
                {maxOutput: 1024},
                mockExec,
            );

            expect(result.code).toBe(1);
            expect(result.asm).toBeUndefined();
            expect(result.stderr).toBe('objdump: test.o: No such file');
        });
    });

    describe('getArgs', () => {
        it('should generate correct arguments', () => {
            const objdumper = new DefaultObjdumper();

            const args = objdumper.getArgs(
                'test.o',
                true, // demangle
                true, // intelAsm
                true, // staticReloc
                false, // dynamicReloc
                ['--custom-arg'],
            );

            expect(args).toContain('-d');
            expect(args).toContain('test.o');
            expect(args).toContain('-l');
            expect(args).toContain('-r'); // staticReloc
            expect(args).not.toContain('-R'); // dynamicReloc is false
            expect(args).toContain('-C'); // demangle
            expect(args).toContain('-M');
            expect(args).toContain('intel'); // intelAsm
            expect(args).toContain('--custom-arg');
        });
    });
});

class TestableCompiler extends BaseCompiler {
    public testGetObjdumperForResult(result: CompilationResult) {
        return this.getObjdumperForResult(result);
    }
}

const languages = {
    'c++': {id: 'c++'},
} as const;

describe('Cross-architecture objdumper selection', () => {
    let ce: ReturnType<typeof makeCompilationEnvironment>;

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    function makeCompiler(overrides: Partial<CompilerInfo>) {
        return new TestableCompiler(
            makeFakeCompilerInfo({
                exe: '/usr/bin/gcc',
                lang: 'c++',
                ldPath: [],
                remote: {target: 'foo', path: 'bar', cmakePath: 'cmake', basePath: '/'},
                objdumper: 'objdump',
                objdumperType: 'default',
                llvmObjdumper: '',
                ...overrides,
            }),
            ce,
        );
    }

    it('should return default objdumper for x86 targets', () => {
        const compiler = makeCompiler({llvmObjdumper: 'llvm-objdump'});
        const result = compiler.testGetObjdumperForResult({instructionSet: 'amd64'} as CompilationResult);
        expect(result).not.toBeNull();
        expect(result!.exe).toBe('objdump');
    });

    it('should return llvm-objdump for non-x86 targets when configured', () => {
        const compiler = makeCompiler({llvmObjdumper: 'llvm-objdump'});
        const result = compiler.testGetObjdumperForResult({instructionSet: 'aarch64'} as CompilationResult);
        expect(result).not.toBeNull();
        expect(result!.exe).toBe('llvm-objdump');
        expect(result!.cls).toBe(LlvmObjdumper);
    });

    it('should return default objdumper for non-x86 targets when llvmObjdumper is not configured', () => {
        const compiler = makeCompiler({llvmObjdumper: ''});
        const result = compiler.testGetObjdumperForResult({instructionSet: 'aarch64'} as CompilationResult);
        expect(result).not.toBeNull();
        expect(result!.exe).toBe('objdump');
    });

    it('should return default objdumper when instructionSet is not set', () => {
        const compiler = makeCompiler({llvmObjdumper: 'llvm-objdump'});
        const result = compiler.testGetObjdumperForResult({} as CompilationResult);
        expect(result).not.toBeNull();
        expect(result!.exe).toBe('objdump');
    });

    it('should return null when no objdumper is configured', () => {
        const compiler = makeCompiler({objdumper: '', objdumperType: ''});
        const result = compiler.testGetObjdumperForResult({instructionSet: 'aarch64'} as CompilationResult);
        expect(result).toBeNull();
    });

    it('should return llvm-objdump for arm targets', () => {
        const compiler = makeCompiler({llvmObjdumper: 'llvm-objdump'});
        const result = compiler.testGetObjdumperForResult({instructionSet: 'arm32'} as CompilationResult);
        expect(result).not.toBeNull();
        expect(result!.exe).toBe('llvm-objdump');
        expect(result!.cls).toBe(LlvmObjdumper);
    });
});
