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

import {afterEach, describe, expect, it, vi} from 'vitest';

import {asmMnemonic, getSimdrefDoc, SIMDREF_BASE_URL, simdrefArch} from '../static/simdref-docs.js';

// Fixtures trimmed from the live https://simdref.diamondinoia.com/detail-chunks/VAD.json and FAD.json
// (2026-10-26), keeping only the fields the client reads.
const vadChunk = {
    'x86:vaddps (xmm, xmm, m128)': {
        summary: 'Add Packed Single Precision Floating-Point Values.',
        display_mnemonic: 'VADDPS',
        display_isa: 'AVX',
        description: {
            Description: 'Adds four packed single precision floating-point values from the second source operand.',
            Operation: 'DEST[31:0] := SRC1[31:0] + SRC2[31:0]',
        },
        measurements: [
            {latency: '4', uarch: 'SKL'},
            {latency: '3', uarch: 'SNB'},
        ],
    },
    'x86:vaddps (xmm, xmm, xmm)': {
        summary: 'Add Packed Single Precision Floating-Point Values.',
        display_mnemonic: 'VADDPS',
        display_isa: 'AVX',
        description: {
            Description: 'Adds four packed single precision floating-point values from the second source operand.',
            Operation: 'DEST[31:0] := SRC1[31:0] + SRC2[31:0]',
        },
        measurements: [{latency: '4', uarch: 'SKL'}],
    },
    'riscv:vadd.vv': {
        summary: 'Vector-vector integer addition.',
        display_mnemonic: 'vadd.vv',
        isa: ['V'],
        description: {
            Description: 'Vector-vector integer addition.',
            Operation: 'vd[i] = vs2[i] + vs1[i]',
        },
        measurements: [],
    },
};

const fadChunk = {
    'arm:fadd (d0, d1, d2)': {
        summary: 'Fadd instruction.',
        display_mnemonic: 'FADD',
        display_isa: 'NEON',
        description: {
            Description: 'Adds the destination and source operands and stores the sum in the destination location.',
            Operation: 'DEST := DEST + SRC;',
        },
        measurements: [{latency: '5', uarch: 'cortex-a72'}],
    },
    'arm:fadd (s0, s1, s2)': {
        summary: 'Fadd instruction.',
        display_mnemonic: 'FADD',
        display_isa: 'NEON',
        description: {
            Description: 'Adds the destination and source operands and stores the sum in the destination location.',
            Operation: 'DEST := DEST + SRC;',
        },
        measurements: [{latency: '5', uarch: 'cortex-a72'}],
    },
    'x86:fadd (st, st(0))': {
        summary: 'Fadd 80-bit operands.',
        display_mnemonic: 'FADD',
        display_isa: 'X87',
        description: {
            Description: 'Adds the destination and source operands and stores the sum in the destination location.',
            Operation: 'DEST := DEST + SRC;',
        },
        measurements: [],
    },
};

function jsonResponse(data: unknown): Response {
    return {ok: true, json: () => Promise.resolve(data)} as unknown as Response;
}

function stubFetch(routes: Record<string, unknown>) {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        for (const [suffix, data] of Object.entries(routes)) {
            if (url.endsWith(suffix)) return Promise.resolve(jsonResponse(data));
        }
        return Promise.reject(new Error(`unexpected fetch ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
}

afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

describe('simdrefArch', () => {
    it('maps CE instruction sets to simdref architectures', () => {
        expect(simdrefArch('amd64')).toBe('x86');
        expect(simdrefArch('x86')).toBe('x86');
        expect(simdrefArch('aarch64')).toBe('arm');
        expect(simdrefArch('arm32')).toBe('arm');
        expect(simdrefArch('riscv32')).toBe('riscv');
        expect(simdrefArch('riscv64')).toBe('riscv');
    });

    it('returns null for unknown or missing instruction sets', () => {
        expect(simdrefArch('6502')).toBeNull();
        expect(simdrefArch(undefined)).toBeNull();
        expect(simdrefArch(null)).toBeNull();
    });
});

describe('asmMnemonic', () => {
    it('parses an Intel x86 line', () => {
        expect(asmMnemonic('        vaddps ymm0, ymm1, ymm2')).toBe('vaddps');
    });

    it('parses an AT&T x86 line', () => {
        expect(asmMnemonic('        addl %edx, %eax')).toBe('addl');
    });

    it('skips a leading x86 lock or rep prefix', () => {
        expect(asmMnemonic('        lock add DWORD PTR [rdi], 1')).toBe('add');
        expect(asmMnemonic('        rep movsb')).toBe('movsb');
        expect(asmMnemonic('        repne scasb')).toBe('scasb');
    });

    it('parses an AArch64 line', () => {
        expect(asmMnemonic('        fadd d0, d1, d2')).toBe('fadd');
    });

    it('parses a RISC-V line', () => {
        expect(asmMnemonic('        vadd.vv v1, v2, v3')).toBe('vadd.vv');
    });

    it('returns null for labels, directives, comments and blank lines', () => {
        expect(asmMnemonic('LBB0_2:')).toBeNull();
        expect(asmMnemonic('        .section .rodata')).toBeNull();
        expect(asmMnemonic('# a comment')).toBeNull();
        expect(asmMnemonic('; a comment')).toBeNull();
        expect(asmMnemonic('// a comment')).toBeNull();
        expect(asmMnemonic('   ')).toBeNull();
    });
});

describe('getSimdrefDoc', () => {
    it('resolves x86 vaddps with summary and url', async () => {
        stubFetch({'/detail-chunks/VAD.json': vadChunk});
        const doc = await getSimdrefDoc('x86', 'vaddps');
        expect(doc?.key).toBe('x86:vaddps (xmm, xmm, m128)');
        expect(doc?.summary.toLowerCase()).toMatch(/^add packed single precision floating-point/);
        expect(doc?.url).toBe(`${SIMDREF_BASE_URL}/#x86%3Avaddps%20(xmm%2C%20xmm%2C%20m128)`);
        expect(doc?.url).toContain('#x86%3Avaddps');
        expect(doc?.markdown).toContain('**VADDPS** (AVX)');
        expect(doc?.markdown).toContain('Latency 3 cycles (SNB)');
        expect(doc?.markdown).toContain(`[Full docs on simdref](${doc?.url})`);
        expect(doc?.markdown).toContain('```');
    });

    it('resolves arm fadd and x86 fadd to different keys', async () => {
        stubFetch({'/detail-chunks/FAD.json': fadChunk});
        const arm = await getSimdrefDoc('arm', 'fadd');
        const x86 = await getSimdrefDoc('x86', 'fadd');
        expect(arm?.key).toBe('arm:fadd (d0, d1, d2)');
        expect(x86?.key).toBe('x86:fadd (st, st(0))');
        expect(arm?.key).not.toBe(x86?.key);
    });

    it('strips the AT&T operand-size suffix once', async () => {
        stubFetch({'/detail-chunks/FAD.json': fadChunk});
        const doc = await getSimdrefDoc('x86', 'faddl');
        expect(doc?.key).toBe('x86:fadd (st, st(0))');
        expect(doc?.summary).toBe('Fadd 80-bit operands.');
    });

    it('returns null when the fetch fails', async () => {
        stubFetch({});
        const doc = await getSimdrefDoc('x86', 'qradd');
        expect(doc).toBeNull();
    });

    it('fetches a chunk only once', async () => {
        const fetchMock = stubFetch({'/detail-chunks/VMA.json': {}});
        await getSimdrefDoc('x86', 'vmaxps');
        await getSimdrefDoc('arm', 'vmaxps');
        expect(fetchMock).toHaveBeenCalledTimes(1);
        expect(String(fetchMock.mock.calls[0][0])).toBe(`${SIMDREF_BASE_URL}/detail-chunks/VMA.json`);
    });

    it('retries the fetch after a failure', async () => {
        const fetchMock = stubFetch({})
            .mockRejectedValueOnce(new Error('boom'))
            .mockResolvedValueOnce(jsonResponse({}));
        expect(await getSimdrefDoc('x86', 'ypadd')).toBeNull();
        expect(await getSimdrefDoc('x86', 'ypadd')).toBeNull();
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });
});
