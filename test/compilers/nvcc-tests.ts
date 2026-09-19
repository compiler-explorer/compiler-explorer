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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {NvccCompiler, NvccWin32Compiler} from '../../lib/compilers/index.js';
import {makeCompilationEnvironment} from '../utils.js';

describe('nvcc tests', () => {
    const languages = {cuda: {id: 'cuda'}};
    const info = {exe: 'foobar', remote: true, lang: 'cuda', ldPath: []};
    const nvcc = new NvccCompiler(info as any, makeCompilationEnvironment({languages}));
    // Access protected method via type cast for testing
    const removeBlob = (asm: string) => (nvcc as any).removeNvccFatbinaryBlob(asm);

    it('removes #APP/#NO_APP block containing .nv_fatbin', () => {
        const asm = `        .text
#APP
        .section .nv_fatbin, "a"
.align 8
fatbinData:
.quad 0x00100001ba55ed50,0x0000000000000fa8
.quad 0x0000000000000000,0x0000003400010007
.text

#NO_APP
        .type   main, @function
main:
        ret
`;
        const result = removeBlob(asm);
        expect(result).not.toContain('fatbinData');
        expect(result).not.toContain('.nv_fatbin');
        expect(result).not.toContain('#APP');
        expect(result).not.toContain('#NO_APP');
        expect(result).not.toContain('.quad 0x00100001');
        expect(result).toContain('main:');
        expect(result).toContain('ret');
    });

    it('preserves #APP/#NO_APP blocks that do NOT contain .nv_fatbin (user inline asm)', () => {
        const asm = `        .text
#APP
        nop
#NO_APP
        ret
`;
        const result = removeBlob(asm);
        expect(result).toContain('#APP');
        expect(result).toContain('nop');
        expect(result).toContain('#NO_APP');
        expect(result).toContain('ret');
    });

    it('handles multiple APP blocks — only removes the fat-binary one', () => {
        const asm = `        .text
#APP
        nop
#NO_APP
        call foo
#APP
        .section .nv_fatbin, "a"
fatbinData:
.quad 0xdeadbeef
#NO_APP
        ret
`;
        const result = removeBlob(asm);
        expect(result).toContain('#APP');
        expect(result).toContain('nop');
        expect(result).not.toContain('fatbinData');
        expect(result).not.toContain('.nv_fatbin');
        expect(result).toContain('call foo');
        expect(result).toContain('ret');
    });

    it('does not strip an APP block where .nv_fatbin appears only in a string, not a section directive', () => {
        // A .string containing ".nv_fatbin" must not be mistaken for the section directive
        const asm = `        .text
#APP
        .string "filename_or_fatbins"
        .string ".nv_fatbin"
#NO_APP
        ret
`;
        const result = removeBlob(asm);
        expect(result).toContain('#APP');
        expect(result).toContain('.nv_fatbin');
        expect(result).toContain('ret');
    });

    it('leaves asm unchanged when there are no #APP blocks', () => {
        const asm = `        .text
main:
        xorl %eax, %eax
        ret
`;
        expect(removeBlob(asm)).toBe(asm);
    });

    it('preserves an unclosed #APP block (malformed input)', () => {
        const asm = `        .text
#APP
        nop
`;
        const result = removeBlob(asm);
        expect(result).toContain('#APP');
        expect(result).toContain('nop');
    });
});

describe('NvccWin32Compiler tests', () => {
    const languages = {cuda: {id: 'cuda'}};
    const info = {exe: 'nvcc.exe', remote: true, lang: 'cuda', ldPath: []};
    let platformDescriptor: PropertyDescriptor;
    let compiler: NvccWin32Compiler;

    beforeEach(() => {
        platformDescriptor = Object.getOwnPropertyDescriptor(process, 'platform')!;
        Object.defineProperty(process, 'platform', {configurable: true, value: 'win32'});
        compiler = new NvccWin32Compiler(info as any, makeCompilationEnvironment({languages}));
    });

    afterEach(() => {
        Object.defineProperty(process, 'platform', platformDescriptor);
        vi.restoreAllMocks();
    });

    it('rejects construction on non-Windows platforms', () => {
        Object.defineProperty(process, 'platform', {configurable: true, value: 'linux'});

        expect(() => new NvccWin32Compiler(info as any, makeCompilationEnvironment({languages}))).toThrow(
            'NvccWin32Compiler can only be used on Windows',
        );
    });

    it('generates options for assembly output', () => {
        expect(
            (compiler as any).optionsForFilter({binary: false, binaryObject: false, execute: false}, 'output.s'),
        ).toEqual([
            '-o',
            'output.obj',
            '-g',
            '-lineinfo',
            '--keep-device-functions',
            '-c',
            '-keep',
            '-keep-dir',
            '.',
            '-Xcompiler',
            '/nologo,/FA,/c,/Faoutput.s,/Fooutput.obj,/Zi,/Fdoutput.pdb',
        ]);
    });

    it('generates options for binary-object output and normalises Windows paths', () => {
        expect(
            (compiler as any).optionsForFilter(
                {binary: false, binaryObject: true, execute: false},
                'C:\\build\\output.obj',
            ),
        ).toEqual([
            '-o',
            'C:\\build\\output.obj',
            '-g',
            '-lineinfo',
            '--keep-device-functions',
            '-c',
            '-keep',
            '-keep-dir',
            '.',
            '-Xcompiler',
            '/nologo,/c,/FoC:/build/output.obj,/Zi',
        ]);
    });

    it('generates options for executable output', () => {
        expect(
            (compiler as any).optionsForFilter(
                {binary: true, binaryObject: false, execute: true},
                'C:\\build\\output.exe',
            ),
        ).toEqual([
            '-o',
            'C:\\build\\output.exe',
            '-g',
            '-lineinfo',
            '--keep-device-functions',
            '-Xcompiler',
            '/nologo,/FeC:/build/output.exe,/Zi',
        ]);
    });
});
