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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import {describe, expect, it} from 'vitest';

import {SPIRVCompiler} from '../lib/compilers/spirv.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {openclc: {id: 'openclc' as LanguageKey}};

describe('SPIR-V compiler', () => {
    it('keeps the bitcode action out of the options it shares with the IR view', async () => {
        const compiler = new SPIRVCompiler(
            makeFakeCompilerInfo({lang: 'openclc' as LanguageKey, exe: '/opt/clang/bin/clang'}),
            makeCompilationEnvironment({languages}),
        );
        const calls: string[][] = [];
        (compiler as any).exec = async (_exe: string, args: string[]) => {
            calls.push([...args]);
            return {code: 1, stdout: '', stderr: '', filenameTransform: (f: string) => f};
        };

        const options = ['-cc1', '-o', '/tmp/dir/output.bc', '/tmp/dir/example.cl'];
        await Promise.all([
            compiler.runCompiler('/opt/clang/bin/clang', options, '/tmp/dir/example.cl', {} as any),
            compiler.generateIR('/tmp/dir/example.cl', options, {} as any, false, {} as any),
        ]);

        expect(options).toEqual(['-cc1', '-o', '/tmp/dir/output.bc', '/tmp/dir/example.cl']);
        const irCall = calls.find(args => args.includes('-emit-llvm'));
        expect(irCall).not.toContain('-emit-llvm-bc');
    });
});
