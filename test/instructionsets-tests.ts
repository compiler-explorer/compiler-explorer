// Copyright (c) 2022, Compiler Explorer Authors
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

import {instructionSetFromTargetString, tripleForInstructionSet} from '../lib/instructionsets.js';

describe('instructionSetFromTargetString', () => {
    it('matches LLVM-style triples to InstructionSet', () => {
        expect(instructionSetFromTargetString('aarch64-linux-gnu')).toBe('aarch64');
        expect(instructionSetFromTargetString('arm-linux-gnueabihf')).toBe('arm32');
        expect(instructionSetFromTargetString('x86_64-pc-windows-msvc')).toBe('amd64');
        expect(instructionSetFromTargetString('riscv64-unknown-linux-gnu')).toBe('riscv64');
        expect(instructionSetFromTargetString('riscv32-unknown-elf')).toBe('riscv32');
        expect(instructionSetFromTargetString('powerpc64le-unknown-linux-gnu')).toBe('powerpc');
        expect(instructionSetFromTargetString('mips64el-unknown-linux-gnuabi64')).toBe('mips');
        expect(instructionSetFromTargetString('s390x-unknown-linux-gnu')).toBe('s390x');
        expect(instructionSetFromTargetString('hppa-unknown-linux-gnu')).toBe('hppa');
        expect(instructionSetFromTargetString('wasm32-unknown-unknown')).toBe('wasm32');
    });

    it('prefers aarch64 over arm32 for AArch64 architecture levels', () => {
        // `-march=armv8-a` etc. are AArch64; the bare `'arm'` substring would
        // mis-match arm32 if the table iteration didn't put aarch64 first.
        expect(instructionSetFromTargetString('armv8-a')).toBe('aarch64');
        expect(instructionSetFromTargetString('armv8.8-a+crypto+sve2')).toBe('aarch64');
        expect(instructionSetFromTargetString('armv9-a')).toBe('aarch64');
        // Bare `arm` (or arm-prefixed triples that aren't armv8/9) are still arm32.
        expect(instructionSetFromTargetString('armv7-a')).toBe('arm32');
        expect(instructionSetFromTargetString('arm')).toBe('arm32');
    });

    it('matches the .NET --targetarch short spellings', () => {
        // ilc / crossgen2 use these short names rather than triples.
        expect(instructionSetFromTargetString('x64')).toBe('amd64');
        expect(instructionSetFromTargetString('arm64')).toBe('aarch64');
        expect(instructionSetFromTargetString('loongarch64')).toBe('loongarch');
        expect(instructionSetFromTargetString('riscv64')).toBe('riscv64');
        expect(instructionSetFromTargetString('wasm')).toBe('wasm32');
        expect(instructionSetFromTargetString('x86')).toBe('x86');
        // wasm64 must still resolve to wasm64 (not wasm32 via the `wasm` alias).
        expect(instructionSetFromTargetString('wasm64')).toBe('wasm64');
    });

    it('recognises classic i386/i486/i586/i686 spellings as x86', () => {
        // GCC/LLVM 32-bit Intel triples don't contain the `x86` substring.
        expect(instructionSetFromTargetString('i386')).toBe('x86');
        expect(instructionSetFromTargetString('i486-pc-linux-gnu')).toBe('x86');
        expect(instructionSetFromTargetString('i586')).toBe('x86');
        expect(instructionSetFromTargetString('i686-pc-linux-gnu')).toBe('x86');
        // amd64 still wins for x86_64 (iterates first; `x86_64` doesn't contain
        // `i386` etc).
        expect(instructionSetFromTargetString('x86_64-pc-linux-gnu')).toBe('amd64');
    });

    it('returns undefined for unrecognised target strings', () => {
        expect(instructionSetFromTargetString('nonsense-target')).toBeUndefined();
        expect(instructionSetFromTargetString('')).toBeUndefined();
    });
});

describe('tripleForInstructionSet', () => {
    it('returns the canonical triple for known InstructionSets', () => {
        expect(tripleForInstructionSet('aarch64')).toBe('aarch64');
        expect(tripleForInstructionSet('amd64')).toBe('x86_64');
        expect(tripleForInstructionSet('riscv64')).toBe('rv64');
        expect(tripleForInstructionSet('powerpc')).toBe('powerpc');
        expect(tripleForInstructionSet('hppa')).toBe('hppa');
        // 32-bit x86 must canonicalise to `i386` so `-mtriple=i386` is what
        // llvm-mca-tool special-cases (`target?.startsWith('i386')`), not the
        // invalid `-mtriple=x86`.
        expect(tripleForInstructionSet('x86')).toBe('i386');
    });

    it('returns null for InstructionSets with no LLVM triple', () => {
        expect(tripleForInstructionSet('python')).toBeNull();
        expect(tripleForInstructionSet('java')).toBeNull();
        expect(tripleForInstructionSet('evm')).toBeNull();
    });

    it('tolerates null/undefined input', () => {
        expect(tripleForInstructionSet(null)).toBeNull();
        expect(tripleForInstructionSet(undefined)).toBeNull();
    });
});
