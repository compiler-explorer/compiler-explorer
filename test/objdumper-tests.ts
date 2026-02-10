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
    rust: {id: 'rust'},
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

describe('getInstructionSetFromCompilerArgs', () => {
    let ce: ReturnType<typeof makeCompilationEnvironment>;

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    function makeCompiler(overrides: Partial<CompilerInfo>) {
        return new TestableCompiler(
            makeFakeCompilerInfo({
                exe: '/usr/bin/rustc',
                lang: 'rust',
                ldPath: [],
                remote: {target: 'foo', path: 'bar', cmakePath: 'cmake', basePath: '/'},
                objdumper: 'objdump',
                objdumperType: 'default',
                llvmObjdumper: 'llvm-objdump',
                ...overrides,
            }),
            ce,
        );
    }

    describe('Rust --target= flag (supportsTargetIs)', () => {
        it('should detect aarch64 from --target=aarch64-unknown-linux-gnu', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs([
                '--edition=2021',
                '--target=aarch64-unknown-linux-gnu',
                '-o',
                'output.s',
            ]);
            expect(iset).toBe('aarch64');
        });

        it('should detect arm32 from --target=arm-unknown-linux-gnueabi', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs([
                '--target=arm-unknown-linux-gnueabi',
                '-o',
                'output.s',
            ]);
            expect(iset).toBe('arm32');
        });

        it('should detect riscv64 from --target=riscv64gc-unknown-linux-gnu', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs([
                '--target=riscv64gc-unknown-linux-gnu',
                '-o',
                'output.s',
            ]);
            expect(iset).toBe('riscv64');
        });

        it('should detect amd64 from --target=x86_64-unknown-linux-gnu', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs([
                '--target=x86_64-unknown-linux-gnu',
                '-o',
                'output.s',
            ]);
            expect(iset).toBe('amd64');
        });

        it('should detect powerpc from --target=powerpc64le-unknown-linux-gnu', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['--target=powerpc64le-unknown-linux-gnu']);
            expect(iset).toBe('powerpc');
        });

        it('should detect mips from --target=mips64el-unknown-linux-gnuabi64', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['--target=mips64el-unknown-linux-gnuabi64']);
            expect(iset).toBe('mips');
        });

        it('should detect s390x from --target=s390x-unknown-linux-gnu', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['--target=s390x-unknown-linux-gnu']);
            expect(iset).toBe('s390x');
        });

        it('should detect wasm32 from --target=wasm32-unknown-unknown', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['--target=wasm32-unknown-unknown']);
            expect(iset).toBe('wasm32');
        });
    });

    describe('Rust --target <value> flag (supportsTarget)', () => {
        it('should detect aarch64 from --target aarch64-unknown-linux-gnu', () => {
            const compiler = makeCompiler({supportsTarget: true});
            const iset = compiler.getInstructionSetFromCompilerArgs([
                '--target',
                'aarch64-unknown-linux-gnu',
                '-o',
                'output.s',
            ]);
            expect(iset).toBe('aarch64');
        });

        it('should detect amd64 from --target x86_64-pc-windows-msvc', () => {
            const compiler = makeCompiler({supportsTarget: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['--target', 'x86_64-pc-windows-msvc']);
            expect(iset).toBe('amd64');
        });
    });

    describe('Zig -target flag (supportsHyphenTarget)', () => {
        it('should detect aarch64 from -target aarch64-linux-gnu', () => {
            const compiler = makeCompiler({exe: '/usr/bin/zig', supportsHyphenTarget: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['build-obj', '-target', 'aarch64-linux-gnu']);
            expect(iset).toBe('aarch64');
        });

        it('should detect arm32 from -target arm-linux-gnueabihf', () => {
            const compiler = makeCompiler({exe: '/usr/bin/zig', supportsHyphenTarget: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['-target', 'arm-linux-gnueabihf']);
            expect(iset).toBe('arm32');
        });

        it('should detect riscv64 from -target riscv64-linux-gnu', () => {
            const compiler = makeCompiler({exe: '/usr/bin/zig', supportsHyphenTarget: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['-target', 'riscv64-linux-gnu']);
            expect(iset).toBe('riscv64');
        });
    });

    describe('GCC -march= flag (supportsMarch)', () => {
        it('should detect aarch64 from -march=aarch64', () => {
            const compiler = makeCompiler({exe: '/usr/bin/gcc', supportsMarch: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['-march=aarch64', '-o', 'output.o']);
            expect(iset).toBe('aarch64');
        });

        it('should detect avr from -march=avr', () => {
            const compiler = makeCompiler({exe: '/usr/bin/avr-gcc', supportsMarch: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['-march=avr', '-o', 'output.o']);
            expect(iset).toBe('avr');
        });
    });

    describe('fallback behaviour', () => {
        it('should default to amd64 when no target flag is present', () => {
            const compiler = makeCompiler({supportsTargetIs: true});
            const iset = compiler.getInstructionSetFromCompilerArgs(['-o', 'output.s', '-O2']);
            expect(iset).toBe('amd64');
        });

        it('should use compiler.instructionSet when no target flag is present', () => {
            const compiler = makeCompiler({supportsTargetIs: true, instructionSet: 'aarch64'});
            const iset = compiler.getInstructionSetFromCompilerArgs(['-o', 'output.s']);
            expect(iset).toBe('aarch64');
        });

        it('should default to amd64 when no target flags are supported', () => {
            const compiler = makeCompiler({});
            const iset = compiler.getInstructionSetFromCompilerArgs(['--target=aarch64-unknown-linux-gnu']);
            expect(iset).toBe('amd64');
        });

        it('should prefer target flag over compiler.instructionSet', () => {
            const compiler = makeCompiler({supportsTargetIs: true, instructionSet: 'arm32'});
            const iset = compiler.getInstructionSetFromCompilerArgs(['--target=aarch64-unknown-linux-gnu']);
            expect(iset).toBe('aarch64');
        });
    });
});

describe('End-to-end: compiler args to objdumper selection', () => {
    let ce: ReturnType<typeof makeCompilationEnvironment>;

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    function makeCompiler(overrides: Partial<CompilerInfo>) {
        return new TestableCompiler(
            makeFakeCompilerInfo({
                exe: '/usr/bin/rustc',
                lang: 'rust',
                ldPath: [],
                remote: {target: 'foo', path: 'bar', cmakePath: 'cmake', basePath: '/'},
                objdumper: 'objdump',
                objdumperType: 'default',
                llvmObjdumper: 'llvm-objdump',
                supportsTargetIs: true,
                ...overrides,
            }),
            ce,
        );
    }

    it('Rust --target=aarch64-unknown-linux-gnu should select llvm-objdump', () => {
        const compiler = makeCompiler({});
        const args = ['--edition=2021', '--target=aarch64-unknown-linux-gnu', '-o', 'output.s', '-S'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('aarch64');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('llvm-objdump');
        expect(objdumperInfo!.cls).toBe(LlvmObjdumper);
    });

    it('Rust --target=x86_64-unknown-linux-gnu should keep default objdump', () => {
        const compiler = makeCompiler({});
        const args = ['--edition=2021', '--target=x86_64-unknown-linux-gnu', '-o', 'output.s'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('amd64');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('objdump');
    });

    it('Rust with no --target flag should keep default objdump (defaults to amd64)', () => {
        const compiler = makeCompiler({});
        const args = ['--edition=2021', '-o', 'output.s', '-S'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('amd64');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('objdump');
    });

    it('Rust --target=arm-unknown-linux-gnueabi should select llvm-objdump', () => {
        const compiler = makeCompiler({});
        const args = ['--target=arm-unknown-linux-gnueabi', '-o', 'output.s'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('arm32');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('llvm-objdump');
        expect(objdumperInfo!.cls).toBe(LlvmObjdumper);
    });

    it('Rust --target=riscv64gc-unknown-linux-gnu should select llvm-objdump', () => {
        const compiler = makeCompiler({});
        const args = ['--target=riscv64gc-unknown-linux-gnu'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('riscv64');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('llvm-objdump');
        expect(objdumperInfo!.cls).toBe(LlvmObjdumper);
    });

    it('Zig -target aarch64-linux-gnu should select llvm-objdump', () => {
        const compiler = makeCompiler({
            exe: '/usr/bin/zig',
            supportsTargetIs: false,
            supportsHyphenTarget: true,
        });
        const args = ['build-obj', '-target', 'aarch64-linux-gnu', 'example.zig'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('aarch64');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('llvm-objdump');
        expect(objdumperInfo!.cls).toBe(LlvmObjdumper);
    });

    it('Rust cross-compile without llvmObjdumper configured falls back to GNU objdump', () => {
        const compiler = makeCompiler({llvmObjdumper: ''});
        const args = ['--target=aarch64-unknown-linux-gnu', '-o', 'output.s'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('aarch64');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('objdump');
    });

    it('Rust --target=s390x-unknown-linux-gnu should select llvm-objdump', () => {
        const compiler = makeCompiler({});
        const args = ['--target=s390x-unknown-linux-gnu'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('s390x');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('llvm-objdump');
        expect(objdumperInfo!.cls).toBe(LlvmObjdumper);
    });

    it('Rust --target=wasm32-unknown-unknown should select llvm-objdump', () => {
        const compiler = makeCompiler({});
        const args = ['--target=wasm32-unknown-unknown'];
        const iset = compiler.getInstructionSetFromCompilerArgs(args);
        expect(iset).toBe('wasm32');

        const objdumperInfo = compiler.testGetObjdumperForResult({instructionSet: iset} as CompilationResult);
        expect(objdumperInfo).not.toBeNull();
        expect(objdumperInfo!.exe).toBe('llvm-objdump');
        expect(objdumperInfo!.cls).toBe(LlvmObjdumper);
    });
});
