// Copyright (c) 2021, Compiler Explorer Authors
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

import {InstructionSet} from '../types/instructionsets.js';

// Mapping between LLVM-style target strings and our InstructionSet enum.
//
// Used in two narrow places:
//   1. `getInstructionSetFromCompilerArgs` — resolves a runtime
//      `-target=foo` / `-march=foo` override to an InstructionSet so the asm
//      view renders the right syntax.
//   2. `tripleForInstructionSet` — picks an `-mtriple=foo` value to feed to
//      llvm-mca.
//
// This is **not** a heuristic for inferring a compiler's default arch; that
// is required to be explicitly declared via `instructionSet=...` in
// properties (see `findCompilersWithoutInstructionSet`). InstructionSets
// that never appear in compiler-target strings (`python`, `evm`, `java`,
// etc.) intentionally have no entry — the asm view stays on whatever the
// compiler's configured arch is.
//
// Each value is a non-empty list of triple substrings that identify the
// InstructionSet. The first entry is also the canonical `-mtriple=` value.
// Substring matching is `String.includes`, so each list omits prefixes
// already covered by an earlier entry (e.g. `'ppc'` covers `ppc64`).
//
// Iteration order is load-bearing: more-specific prefixes must come before
// strict-substring ones (e.g. `aarch64` before `arm32`, `wasm64` before
// `wasm32`). The TypeScript-preserved declaration order is what
// `Object.entries` walks.
const TARGET_SUBSTRINGS: Partial<Record<InstructionSet, readonly string[]>> = {
    // AArch64: `armv8`/`armv9` are AArch64 architecture levels (`-march=armv8-a`).
    // `arm64` is the .NET `--targetarch` spelling. Listed before `arm32`'s `'arm'`.
    aarch64: ['aarch64', 'armv8', 'armv9', 'arm64'],
    // `x64` is the .NET `--targetarch` spelling for amd64.
    amd64: ['x86_64', 'x64'],
    arm32: ['arm'],
    avr: ['avr'],
    c6x: ['tic6x'],
    ebpf: ['bpf'],
    ez80: ['ez80'],
    hppa: ['hppa'],
    kvx: ['kvx'],
    loongarch: ['loongarch'],
    m68k: ['m68k'],
    mips: ['mips'],
    msp430: ['msp430'],
    powerpc: ['powerpc', 'ppc'],
    riscv32: ['rv32', 'riscv32'],
    riscv64: ['rv64', 'riscv64'],
    s390x: ['s390x'],
    sh: ['sh'],
    sparc: ['sparc'],
    vax: ['vax'],
    // wasm64 must be checked before wasm32, otherwise the bare `wasm` alias
    // (added below for .NET `--targetarch wasm`) would steal `wasm64-...`.
    wasm64: ['wasm64'],
    wasm32: ['wasm32', 'wasm'],
    // 32-bit Intel: classic LLVM/GCC triples spell the family `i386`/`i486`/
    // `i586`/`i686` (no `x86` substring at all). The canonical `-mtriple=`
    // value is `i386` — what `lib/tooling/llvm-mca-tool.ts` already
    // special-cases with `target?.startsWith('i386')`. The bare `x86` alias
    // catches `-march=x86` and the .NET `--targetarch x86`.
    x86: ['i386', 'i486', 'i586', 'i686', 'x86'],
    xtensa: ['xtensa'],
    z180: ['z180'],
    z80: ['z80'],
};

// Returns the InstructionSet identified by a compiler-target string
// (e.g. "aarch64-unknown-linux-gnu" → "aarch64"), or undefined if no known
// InstructionSet matches.
export function instructionSetFromTargetString(target: string): InstructionSet | undefined {
    for (const [iset, substrings] of Object.entries(TARGET_SUBSTRINGS) as [InstructionSet, readonly string[]][]) {
        for (const substring of substrings) {
            if (target.includes(substring)) return iset;
        }
    }
    return undefined;
}

// Returns the canonical `-mtriple=` string for an InstructionSet, or null
// if the InstructionSet has no associated LLVM target triple (typical for
// VM/IR formats like `python`, `java`, `evm`). Tolerates null/undefined
// to ease use at call sites where the compiler's instructionSet may be
// nullable in the type even though config validation guarantees a value.
export function tripleForInstructionSet(instructionSet: InstructionSet | null | undefined): string | null {
    if (!instructionSet) return null;
    return TARGET_SUBSTRINGS[instructionSet]?.[0] ?? null;
}
