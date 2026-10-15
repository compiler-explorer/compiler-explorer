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

// Links to ABI Explorer (https://abiexplorer.org/), which runs clang in the
// browser and shows how a record is laid out for a given target.
//
// A whole ABI Explorer session lives in the URL fragment, so a link is built
// entirely on the client and the source never leaves the browser. The wire
// format and its encodings belong to ABI Explorer's own share-link package;
// what is decided here is what CE knows about the code: its language, and the
// standard and target of each compiler attached to it.
//
// ABI Explorer validates every value it reads: an unknown standard becomes the
// language's default, an empty target its default target, and a target clang
// rejects is shown with clang's diagnostics. So anything left blank here is
// safe, and a guess that is wrong is one click from right.

import {
    encode,
    encodePlain,
    MAX_BUFFER_NAME,
    MAX_BUFFERS,
    type ShareBuffer,
    type ShareState,
    toWireV3,
} from '@ambrus-toth/abi-explorer-share-link';

export type AbiExplorerLanguage = 'c' | 'c++' | 'hylo';

export const ABI_EXPLORER_URL = 'https://abiexplorer.org/';

export interface AbiExplorerCompiler {
    name: string;
    /** Already normalised; see `normalizeAbiExplorerStd`. Empty for the language default. */
    std: string;
    /** Empty for ABI Explorer's default target. */
    triple: string;
}

/**
 * Draft spellings and older names, folded onto the released names ABI Explorer
 * lists: it would otherwise replace `-std=c++2b` with its default.
 */
const STD_ALIASES: Record<string, string> = {
    c90: 'c89',
    gnu90: 'gnu89',
    c9x: 'c99',
    gnu9x: 'gnu99',
    c1x: 'c11',
    gnu1x: 'gnu11',
    c18: 'c17',
    gnu18: 'gnu17',
    c2x: 'c23',
    gnu2x: 'gnu23',
    'c++98': 'c++03',
    'gnu++98': 'gnu++03',
    'c++0x': 'c++11',
    'gnu++0x': 'gnu++11',
    'c++1y': 'c++14',
    'gnu++1y': 'gnu++14',
    'c++1z': 'c++17',
    'gnu++1z': 'gnu++17',
    'c++2a': 'c++20',
    'gnu++2a': 'gnu++20',
    'c++2b': 'c++23',
    'gnu++2b': 'gnu++23',
    'c++2c': 'c++26',
    'gnu++2c': 'gnu++26',
};

/**
 * Sanity checks for a triple, by the instruction set CE reports for the
 * compiler, plus a default for when the compiler names no triple at all. The
 * entries without a default cover several ABIs at once (32- and 64-bit, big-
 * and little-endian), where picking one would be a coin toss.
 */
const TARGETS: Record<string, {triple?: string; arch: RegExp}> = {
    amd64: {triple: 'x86_64-unknown-linux-gnu', arch: /^(x86_64|amd64)/},
    x86: {triple: 'i386-unknown-linux-gnu', arch: /^(i[3-6]86|x86)$/},
    aarch64: {triple: 'aarch64-unknown-linux-gnu', arch: /^(aarch64|arm64)/},
    arm32: {triple: 'arm-unknown-linux-gnueabihf', arch: /^(arm|thumb)(?!64)/},
    riscv32: {triple: 'riscv32-unknown-elf', arch: /^riscv32/},
    riscv64: {triple: 'riscv64-unknown-linux-gnu', arch: /^riscv64/},
    s390x: {triple: 's390x-unknown-linux-gnu', arch: /^s390x/},
    loongarch: {triple: 'loongarch64-unknown-linux-gnu', arch: /^loongarch/},
    wasm32: {triple: 'wasm32-unknown-unknown', arch: /^wasm32/},
    wasm64: {triple: 'wasm64-unknown-unknown', arch: /^wasm64/},
    powerpc: {arch: /^(powerpc|ppc)/},
    mips: {arch: /^mips/},
    sparc: {arch: /^sparc/},
};

/** What ABI Explorer accepts as a triple; anything else it reads as empty. */
const TRIPLE_RE = /^[A-Za-z0-9_.-]{1,64}$/;

const CONFIGURED_TARGET_RE = /(?:^|\s)--?target[=\s]\s*(\S+)/;

const VERSION_TARGET_RE = /^Target:\s*(\S+)/m;

/**
 * The triple a compiler builds for, from what CE knows about it without
 * compiling anything:
 *
 * 1. The `-target` in the compiler's configured flags. CE's fleet is largely
 *    one clang binary entered many times with a different target each time,
 *    and this flag is the only exact statement of it.
 * 2. The target the binary defaults to, from its version banner: the real
 *    answer for a distributed cross toolchain such as MinGW's clang.
 * 3. A default for the instruction set CE reports.
 *
 * Where CE's instruction set is one we know, a candidate must agree with it to
 * be believed: a host clang's banner names the host, whatever it is configured
 * for. Where it is not, the candidate is passed on as it is.
 *
 * Undefined where nothing is known well enough to say. The user's own flags
 * are not read: for most cross compilers the target is which binary runs, and
 * appears in no argument.
 */
export function abiExplorerTripleFor(
    instructionSet: string | null | undefined,
    fullVersion?: string,
    compilerOptions?: string,
): string | undefined {
    const known = instructionSet ? TARGETS[instructionSet] : undefined;
    const believable = (target: string | undefined): target is string =>
        !!target && TRIPLE_RE.test(target) && (!known || known.arch.test(target));
    const configured = compilerOptions ? CONFIGURED_TARGET_RE.exec(compilerOptions)?.[1] : undefined;
    if (believable(configured)) return configured;
    const printed = fullVersion ? VERSION_TARGET_RE.exec(fullVersion)?.[1] : undefined;
    if (believable(printed)) return printed;
    return known?.triple;
}

export function abiExplorerLanguageFor(languageId: string | undefined): AbiExplorerLanguage | null {
    return languageId === 'c' || languageId === 'c++' || languageId === 'hylo' ? languageId : null;
}

/** Accepts the flag as written (`-std=gnu++2b`) or just its value. */
export function normalizeAbiExplorerStd(std: string): string {
    const value = std
        .trim()
        .replace(/^-std=/, '')
        .toLowerCase();
    return STD_ALIASES[value] ?? value;
}

function makeBuffer(source: string, lang: AbiExplorerLanguage, compiler?: AbiExplorerCompiler): ShareBuffer {
    return {
        name: (compiler?.name ?? '').slice(0, MAX_BUFFER_NAME),
        source,
        options: {
            lang,
            std: compiler?.std ?? '',
            triple: compiler?.triple ?? '',
            pack: '',
            msBitfields: false,
            shortEnums: false,
            shortWchar: false,
            warnPadded: false,
            extraFlags: '',
        },
        selectedRecord: null,
    };
}

/**
 * One buffer per distinct way the code is built, so an editor with a compiler
 * for each of several targets opens with their layouts side by side. Compilers
 * that agree on standard and target would show the same layout, and share a
 * buffer.
 */
export function abiExplorerState(
    source: string,
    lang: AbiExplorerLanguage,
    compilers: AbiExplorerCompiler[],
): ShareState {
    const distinct = new Map<string, AbiExplorerCompiler>();
    for (const compiler of compilers) {
        const key = `${compiler.std}\n${compiler.triple}`;
        if (!distinct.has(key)) distinct.set(key, compiler);
    }
    const buffers = [...distinct.values()].slice(0, MAX_BUFFERS).map(c => makeBuffer(source, lang, c));
    return {buffers: buffers.length > 0 ? buffers : [makeBuffer(source, lang)], view: 'tabs'};
}

export async function getAbiExplorerUrl(state: ShareState): Promise<string> {
    return `${ABI_EXPLORER_URL}#${await encode(state)}`;
}

/** As `getAbiExplorerUrl`, without waiting, and so without the compression. */
export function getAbiExplorerUrlSync(state: ShareState): string {
    return `${ABI_EXPLORER_URL}#${encodePlain(toWireV3(state))}`;
}
