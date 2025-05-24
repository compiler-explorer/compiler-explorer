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

import {describe, expect, it} from 'vitest';

import {DefaultObjdumper} from '../lib/objdumper/default.js';

import type {ExecutionOptions} from '../types/compilation/compilation.interfaces.js';
import type {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';

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

    describe('getDefaultArgs', () => {
        it('should generate correct arguments', () => {
            const objdumper = new DefaultObjdumper();

            const args = objdumper.getDefaultArgs(
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
