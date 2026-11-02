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

import {beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {TornadoVmTcCompiler} from '../lib/compilers/index.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {makeCompilationEnvironment} from './utils.js';

const languages = {
    java: {id: 'java'},
};

const info = {
    exe: null,
    remote: true,
    lang: languages.java.id,
} as unknown as CompilerInfo;

describe('TornadoVM TC', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    it('maps each statement to the Java line in its trailing comment', async () => {
        const compiler = new TornadoVmTcCompiler(info, env);
        const cuda = [
            '// ===== Saxpy.saxpy =====',
            'extern "C" __global__ void Saxpy_saxpy(float alpha) {',
            '  i = (int) (blockIdx.x * blockDim.x + threadIdx.x);',
            '    y_data[v6] = fmaf(alpha, x_data[v6], y_data[v6]);  // Saxpy.java:9',
            '}',
            '',
        ].join('\n');
        const result = await compiler.processAsm({asm: cuda});
        expect(result.asm).toEqual([
            {text: '// ===== Saxpy.saxpy =====', source: null},
            {text: 'extern "C" __global__ void Saxpy_saxpy(float alpha) {', source: null},
            {text: '  i = (int) (blockIdx.x * blockDim.x + threadIdx.x);', source: null},
            {text: '    y_data[v6] = fmaf(alpha, x_data[v6], y_data[v6]);', source: {file: null, line: 9}},
            {text: '}', source: null},
        ]);
    });

    it('passes an error document through', async () => {
        const compiler = new TornadoVmTcCompiler(info, env);
        expect((await compiler.processAsm({asm: '<No output file>'})).asm).toEqual([
            {text: '<No output file>', source: null},
        ]);
    });

    it('gives the target options to TC and the rest to javac', () => {
        const compiler = new TornadoVmTcCompiler(info, env);
        expect(compiler.splitOptions(['--cc', '90', '-Xlint:all', '--fp16', 'off'])).toEqual({
            tc: ['--cc', '90', '--fp16', 'off'],
            javac: ['-Xlint:all'],
        });
    });

    it('reports success when any kernel compiled, so one refusal does not hide the others', () => {
        const compiler = new TornadoVmTcCompiler(info, env);
        const refused = '// ===== K.a =====\n// K.java:3: error: TC-E4010 TileContext kernels ...\n\n';
        const compiled = '// ===== K.b =====\nextern "C" __global__ void K_b() {\n}\n';
        expect(compiler.compiledAnyKernel(refused)).toBe(false);
        expect(compiler.compiledAnyKernel(refused + compiled)).toBe(true);
    });

    it('passes --arg to TC', () => {
        const compiler = new TornadoVmTcCompiler(info, env);
        expect(compiler.splitOptions(['--arg', 'n=1024', '-Xlint:all']).tc).toEqual(['--arg', 'n=1024']);
    });

    it('never asks for javap', () => {
        const compiler = new TornadoVmTcCompiler(info, env);
        const filters = {binary: true} as any;
        compiler.optionsForFilter(filters);
        expect(filters.binary).toBe(false);
    });
});
