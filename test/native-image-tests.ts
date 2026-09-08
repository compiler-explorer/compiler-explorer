// Copyright (c) 2026, Compiler Explorer Authors
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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import fs from 'node:fs/promises';
import path from 'node:path';

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {NativeImageCompiler, parseNativeImageDisassembly} from '../lib/compilers/native-image.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const method = {
    name: 'Square.square(int)int',
    code: '0fafffc3',
    patches: [],
    mappings: [{start: 0, end: 3, line: 2}],
};
const disassembly = '   0:  0f af ff    imul   %edi,%edi\n   3:  c3          ret\n';
const bytecode =
    'Compiled from "example.java"\nclass Square {\n  static int square(int);\n    Code:\n       0: iload_0\n       1: iload_0\n       2: imul\n       3: ireturn\n    LineNumberTable:\n      line 2: 0\n}\n';

function execResult(stdout = '', code = 0, stderr = '', timedOut = false): UnprocessedExecResult {
    return {
        stdout,
        stderr,
        code,
        timedOut,
        truncated: false,
        okToCache: !timedOut,
        filenameTransform: filename => filename,
        execTime: 1,
    };
}

describe('Native Image disassembly', () => {
    it('associates native instructions with source lines', () => {
        const asm = parseNativeImageDisassembly(method, disassembly);
        expect(asm[1]).toMatchObject({
            text: '    imul   %edi,%edi',
            source: {file: null, line: 2},
            opcodes: ['0f', 'af', 'ff'],
        });
        expect(asm[2].source).toBeNull();
    });

    it('replaces unrelocated calls and conditional runtime branches with symbolic targets', () => {
        const asm = parseNativeImageDisassembly(
            {...method, patches: [{offset: 0, kind: 'call', target: 'Runtime.throw()void'}]},
            '  0: 0f 86 00 00 00 00   jbe 0x6\n',
        );
        expect(asm[1].text).toBe('    jbe <Runtime.throw()void>');
        expect(asm[1].text).not.toContain('0x6');
    });

    it('retains indirect call operands', () => {
        const asm = parseNativeImageDisassembly(
            {...method, patches: [{offset: 0, kind: 'call', target: 'Handler.run()void', direct: false}]},
            '  0: ff d0   call rax\n',
        );
        expect(asm[1].text).toBe('    call rax # indirect call Handler.run()void');
    });

    it('labels branches within a method', () => {
        const asm = parseNativeImageDisassembly(method, '  0: 74 01   je 0x3\n  2: 90   nop\n  3: c3   ret\n');
        const target = asm[1].text.trim().split(' ').at(-1);
        expect(asm.some(line => line.text === `${target}:`)).toBe(true);
        expect(target).toMatch(/^\.L/);
    });

    it('annotates data relocations inside an instruction and removes misleading addresses', () => {
        const asm = parseNativeImageDisassembly(
            {...method, patches: [{offset: 3, kind: 'data', target: 'constant'}]},
            '  0: 48 8d 05 00 00 00 00   lea 0x0(%rip),%rax # 0x7\n',
        );
        expect(asm[1].text).toContain('# unresolved constant');
        expect(asm[1].text).not.toContain('# 0x7');
    });
});

describe('Native Image compiler pipeline', () => {
    let compiler: NativeImageCompiler;
    const directory = path.resolve('/tmp/native-image-test');
    const input = path.join(directory, 'example.java');

    beforeEach(() => {
        compiler = new NativeImageCompiler(
            makeFakeCompilerInfo({
                id: 'native',
                exe: '/graal/bin/native-image',
                lang: 'java',
                objdumper: '/usr/bin/objdump',
                options: '',
            }),
            makeCompilationEnvironment({
                languages: {java: {id: 'java'}},
                props: {
                    'compiler.native.frontend': '/graal/bin/javac',
                    'compiler.native.nativeImageFeature': '/feature.jar',
                },
            }),
        );
        vi.spyOn(fs, 'mkdir').mockResolvedValue(undefined);
        vi.spyOn(fs, 'readdir').mockResolvedValue(['Square.class'] as never);
        vi.spyOn(fs, 'stat').mockResolvedValue({size: 500} as Awaited<ReturnType<typeof fs.stat>>);
        vi.spyOn(fs, 'readFile').mockResolvedValue(
            JSON.stringify({version: 1, architecture: 'amd64', methods: [method]}),
        );
        vi.spyOn(fs, 'writeFile').mockResolvedValue(undefined);
    });

    afterEach(() => vi.restoreAllMocks());

    it('returns bytecode and native assembly from one frontend compilation', async () => {
        const exec = vi.spyOn(compiler, 'exec').mockImplementation(async exe => {
            if (exe.endsWith('javap')) return execResult(bytecode);
            if (exe.endsWith('objdump')) return execResult(disassembly);
            return execResult();
        });
        const result = await compiler.runCompiler(
            compiler.compiler.exe,
            ['-O2'],
            input,
            compiler.getDefaultExecOptions(),
        );
        expect(result.code).toBe(0);
        expect(result.jvmBytecodeOutput).toEqual(
            expect.arrayContaining([expect.objectContaining({text: expect.stringContaining('imul')})]),
        );
        expect(result.asm).toEqual(expect.arrayContaining([expect.objectContaining({text: '    imul   %edi,%edi'})]));
        expect(exec.mock.calls.filter(call => call[0].endsWith('javac'))).toHaveLength(1);
        expect(exec.mock.calls.find(call => call[0].endsWith('native-image'))?.[1]).toContain(
            '--features=ce.nativeimage.ExplorerFeature',
        );
        expect(result.stdout.some(line => line.text.includes('iload'))).toBe(false);
    });

    it('stops immediately on frontend diagnostics', async () => {
        const exec = vi.spyOn(compiler, 'exec').mockResolvedValue(execResult('', 1, 'syntax error'));
        const result = await compiler.runCompiler(compiler.compiler.exe, [], input, compiler.getDefaultExecOptions());
        expect(result.code).toBe(1);
        expect(exec).toHaveBeenCalledTimes(1);
        expect(result.stderr).toEqual(expect.arrayContaining([expect.objectContaining({text: 'syntax error'})]));
    });

    it('keeps bytecode when native compilation times out', async () => {
        vi.spyOn(compiler, 'exec').mockImplementation(async exe =>
            exe.endsWith('javap')
                ? execResult(bytecode)
                : exe.endsWith('native-image')
                  ? execResult('', -1, 'Timed out', true)
                  : execResult(),
        );
        const result = await compiler.runCompiler(compiler.compiler.exe, [], input, compiler.getDefaultExecOptions());
        expect(result.timedOut).toBe(true);
        expect(result.okToCache).toBe(false);
        expect(result.jvmBytecodeOutput?.length).toBeGreaterThan(0);
    });

    it('requires a completion manifest even when native-image exits successfully', async () => {
        vi.spyOn(compiler, 'exec').mockResolvedValue(execResult());
        vi.mocked(fs.stat).mockRejectedValue(new Error('ENOENT'));
        const result = await compiler.runCompiler(compiler.compiler.exe, [], input, compiler.getDefaultExecOptions());
        expect(result.code).toBe(-1);
        expect(result.okToCache).toBe(false);
        expect(result.stderr.at(-1)?.text).toContain('extraction failed');
    });

    it('rejects unsupported architectures instead of disassembling with the wrong instruction set', async () => {
        vi.spyOn(compiler, 'exec').mockResolvedValue(execResult());
        vi.mocked(fs.readFile).mockResolvedValue(
            JSON.stringify({version: 1, architecture: 'aarch64', methods: [method]}),
        );
        const result = await compiler.runCompiler(compiler.compiler.exe, [], input, compiler.getDefaultExecOptions());
        expect(result.code).toBe(-1);
    });

    it('rejects builder features and output overrides', async () => {
        const exec = vi.spyOn(compiler, 'exec');
        const result = await compiler.runCompiler(
            compiler.compiler.exe,
            ['--features=Untrusted'],
            input,
            compiler.getDefaultExecOptions(),
        );
        expect(result.code).toBe(-1);
        expect(exec).not.toHaveBeenCalled();
    });
});
