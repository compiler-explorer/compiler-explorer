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
// entirely on the client: there is no API to call, and the source never leaves
// the browser. Two encodings exist, and the site reads both:
//
//     #2.<base64url(deflate-raw(json))>   what it writes itself
//     #<base64url(json)>                  the original, still accepted
//
// The JSON uses short keys (see `Wire`). ABI Explorer validates every field on
// the way in and silently substitutes its own default for anything it does not
// recognise, so an unknown standard or triple opens the page on defaults rather
// than failing. The format lives in src/core/url-state.ts of
// https://github.com/tothambrus11/abi-explorer-2.
//
// The fragment is read once, when the page loads: a link must therefore be
// opened in a fresh tab, not by rewriting the hash of one already showing
// ABI Explorer.

/** The languages ABI Explorer lays out. Its ids happen to match ours for C and C++. */
export type AbiExplorerLanguage = 'c' | 'c++' | 'hylo';
/** `-fpack-struct=N`, as ABI Explorer spells it; the empty string is the target default. */
export type AbiExplorerPack = '' | '1' | '2' | '4' | '8' | '16';
/** One record at a time behind a tab bar, or all of them stacked. */
export type AbiExplorerView = 'tabs' | 'stack';

export const ABI_EXPLORER_URL = 'https://abiexplorer.org/';

/** What ABI Explorer lays out for when the link does not say. */
export const ABI_EXPLORER_DEFAULT_TRIPLE = 'x86_64-unknown-linux-gnu';

/**
 * The ABI Hylo lays types out for.
 *
 * A name rather than a choice: its compiler describes exactly one, so ABI
 * Explorer hides the target selector for Hylo and lays out for this whatever
 * the link says.
 */
export const ABI_EXPLORER_HYLO_TRIPLE = 'hylo';

/** Their validation of the `t` field: anything else is replaced by the default triple. */
const TRIPLE_RE = /^[\w.-]{1,64}$/;

/** Their cap on the free-form flag string. */
const MAX_EXTRA_FLAGS = 500;

/** Standards the selector offers for C; anything else falls back to their newest. */
export const ABI_EXPLORER_C_STANDARDS: readonly string[] = [
    'c89',
    'gnu89',
    'c99',
    'gnu99',
    'c11',
    'gnu11',
    'c17',
    'gnu17',
    'c23',
    'gnu23',
];

/** Standards the selector offers for C++. */
export const ABI_EXPLORER_CXX_STANDARDS: readonly string[] = [
    'c++03',
    'gnu++03',
    'c++11',
    'gnu++11',
    'c++14',
    'gnu++14',
    'c++17',
    'gnu++17',
    'c++20',
    'gnu++20',
    'c++23',
    'gnu++23',
    'c++26',
    'gnu++26',
];

/**
 * Draft spellings and older names for the standards above.
 *
 * Users write `-std=c++2b` as readily as `-std=c++23`, and ABI Explorer lists
 * only the released name, so the draft ones are folded onto it here rather than
 * dropped.
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
 * An ABI Explorer session, in our own spelling.
 *
 * Everything but the source and the language is optional: an omitted field
 * means "whatever ABI Explorer starts on", which is also what it does with a
 * value it rejects. That is what lets a caller fill in only the parts it knows
 * about, and what will let the target and the layout options below be derived
 * from the compiler's architecture and settings later on.
 */
export interface AbiExplorerState {
    source: string;
    lang: AbiExplorerLanguage;
    /** A standard from the lists above; see `normalizeAbiExplorerStd`. */
    std?: string;
    /** The target to lay out for. Any triple clang accepts, not just the listed ones. */
    triple?: string;
    pack?: AbiExplorerPack;
    msBitfields?: boolean;
    shortEnums?: boolean;
    shortWchar?: boolean;
    warnPadded?: boolean;
    /**
     * Extra clang flags. ABI Explorer allows only flags that change layout
     * (`-f…`, `-m…`, `-W…`, `-D`, `-U`, `-O…`, absolute `-I`/`-isystem`, …) and
     * drops the rest, `-target` among them: the target is `triple`.
     */
    extraFlags?: string;
    /** Record to open on, by ABI Explorer's own key; null means it chooses. */
    selectedRecord?: string | null;
    view?: AbiExplorerView;
}

/** The state as it goes into the fragment: short keys, because each byte is a character of the link. */
interface Wire {
    v: number;
    s: string;
    l: AbiExplorerLanguage;
    std: string;
    t: string;
    p: AbiExplorerPack;
    mb: number;
    se: number;
    sw: number;
    wp: number;
    x: string;
    r: string | null;
    vw: AbiExplorerView;
}

function toWire(state: AbiExplorerState): Wire {
    const triple = state.triple ?? defaultTripleFor(state.lang);
    return {
        v: 2,
        s: state.source,
        l: state.lang,
        std: normalizeAbiExplorerStd(state.lang, state.std ?? ''),
        t: TRIPLE_RE.test(triple) ? triple : ABI_EXPLORER_DEFAULT_TRIPLE,
        p: state.pack ?? '',
        mb: state.msBitfields ? 1 : 0,
        se: state.shortEnums ? 1 : 0,
        sw: state.shortWchar ? 1 : 0,
        wp: state.warnPadded ? 1 : 0,
        x: (state.extraFlags ?? '').slice(0, MAX_EXTRA_FLAGS),
        r: state.selectedRecord ?? null,
        vw: state.view ?? 'tabs',
    };
}

/**
 * Targets we are confident about, by the instruction set CE reports.
 *
 * `triple` is what to lay out for when nothing more exact is known. The
 * entries without one are instruction sets that cover several ABIs at once —
 * 32- and 64-bit, big- and little-endian — where picking a default would be a
 * coin toss. `arch` matches the architecture an exact triple starts with, and
 * is what decides whether that triple and this instruction set are describing
 * the same machine.
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

/** The target CE aims a compiler at, in the flags its configuration gives it. */
const CONFIGURED_TARGET_RE = /(?:^|\s)--?target[=\s]\s*(\S+)/;

/** The target a compiler was built to default to, as its version banner reports it. */
const VERSION_TARGET_RE = /^Target:\s*(\S+)/m;

/**
 * The triple to lay out for, from what CE knows about a compiler without
 * compiling anything.
 *
 * In order of what each source actually claims:
 *
 * 1. The `-target` in the compiler's configured flags. Every clang is a cross
 *    compiler, so CE's fleet is largely one clang binary entered many times
 *    with a different target each time; this flag is the entry's whole point
 *    and the only exact statement of it.
 * 2. Failing that, the target the binary defaults to, from its banner. This
 *    means something only when nothing above overrode it — for a distributed
 *    cross toolchain like MinGW's clang it is the real answer, and for a
 *    host clang it is the host.
 * 3. Failing that, a default for the instruction set CE reports.
 *
 * Each candidate must agree with that instruction set to be believed, since
 * the instruction set is CE's own conclusion about the compiler. A bare
 * architecture (`wasm32`) is passed over for the mapped triple: clang and ABI
 * Explorer both take it, but the fuller spelling is the one ABI Explorer's
 * target list names.
 *
 * Undefined where nothing is known well enough to say, which leaves the link
 * on ABI Explorer's own default.
 */
export function abiExplorerTripleFor(
    instructionSet: string | null | undefined,
    fullVersion?: string,
    compilerOptions?: string,
): string | undefined {
    const known = instructionSet ? TARGETS[instructionSet] : undefined;
    if (!known) return undefined;
    const believable = (target: string | undefined): target is string =>
        !!target && target.includes('-') && known.arch.test(target);
    const configured = compilerOptions ? CONFIGURED_TARGET_RE.exec(compilerOptions)?.[1] : undefined;
    if (believable(configured)) return configured;
    const printed = fullVersion ? VERSION_TARGET_RE.exec(fullVersion)?.[1] : undefined;
    if (believable(printed)) return printed;
    return known.triple;
}

/** The target a language lays out for when the caller does not name one. */
function defaultTripleFor(lang: AbiExplorerLanguage): string {
    return lang === 'hylo' ? ABI_EXPLORER_HYLO_TRIPLE : ABI_EXPLORER_DEFAULT_TRIPLE;
}

/**
 * The ABI Explorer language for one of our language ids, or null for a language
 * it cannot lay out — which is also the answer to "should the button be shown?".
 *
 * Our ids for the three it knows are its own, so this is a membership test.
 */
export function abiExplorerLanguageFor(languageId: string | undefined): AbiExplorerLanguage | null {
    return languageId === 'c' || languageId === 'c++' || languageId === 'hylo' ? languageId : null;
}

/**
 * The standard ABI Explorer would name for this one, or the empty string when
 * there is none, which leaves it on its own default.
 *
 * Accepts the flag as written (`-std=gnu++2b`) or just its value, and folds
 * draft spellings onto the released name.
 */
export function normalizeAbiExplorerStd(lang: AbiExplorerLanguage, std: string): string {
    // Hylo has had one release and no standards to choose between.
    if (lang === 'hylo') return '';
    const value = std
        .trim()
        .replace(/^-std=/, '')
        .toLowerCase();
    if (!value) return '';
    const canonical = STD_ALIASES[value] ?? value;
    const known = lang === 'c++' ? ABI_EXPLORER_CXX_STANDARDS : ABI_EXPLORER_C_STANDARDS;
    return known.includes(canonical) ? canonical : '';
}

/** base64url: base64 with the URL-safe alphabet and no padding, so a link stays a link. */
function base64Url(bytes: Uint8Array): string {
    // Chunked, because spreading a long source into `fromCharCode` overflows the stack.
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

async function deflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
    const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

function encodeUncompressed(wire: Wire): string {
    return base64Url(new TextEncoder().encode(JSON.stringify({...wire, v: 1})));
}

/**
 * Encodes a session into a fragment value, without the leading '#'.
 *
 * Compressed where the browser can compress, which matters because the fragment
 * carries the whole source; the uncompressed encoding is the fallback, and
 * ABI Explorer reads both.
 */
export async function encodeAbiExplorerFragment(state: AbiExplorerState): Promise<string> {
    const wire = toWire(state);
    if (typeof CompressionStream !== 'undefined') {
        try {
            const json = new TextEncoder().encode(JSON.stringify(wire));
            return `2.${base64Url(await deflateRaw(json))}`;
        } catch {
            // Fall through to the uncompressed encoding below.
        }
    }
    return encodeUncompressed(wire);
}

/**
 * The same fragment, uncompressed, for a caller that cannot wait: a link on a
 * button has to be right at the moment it is clicked.
 */
export function encodeAbiExplorerFragmentSync(state: AbiExplorerState): string {
    return encodeUncompressed(toWire(state));
}

/** A link that opens ABI Explorer on this session. */
export async function getAbiExplorerUrl(state: AbiExplorerState): Promise<string> {
    return `${ABI_EXPLORER_URL}#${await encodeAbiExplorerFragment(state)}`;
}

/** As `getAbiExplorerUrl`, without waiting, and so without the compression. */
export function getAbiExplorerUrlSync(state: AbiExplorerState): string {
    return `${ABI_EXPLORER_URL}#${encodeAbiExplorerFragmentSync(state)}`;
}
