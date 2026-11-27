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

import {ClientState} from '../clientstate.js';
import {QUICK_BENCH_DEFAULT_COMPILER, type QuickBenchCandidates, resolveQuickBenchCompiler} from './compilers.js';
import {stripQuickBenchHashBoilerplate, unwrapQuickBenchCode, wrapQuickBenchCode} from './source.js';

// Turns a Quick Bench benchmark into CE client state: one C++ editor holding the benchmark with Quick Bench's hidden
// wrapper made visible, and one executor with the matching compiler, options and libraries. CE has no benchmark view
// yet, so an executor (with Google Benchmark selected) stands in for it.

/** The options Quick Bench needs to build a benchmark, after normalising its .opt files. */
export type QuickBenchOptions = {
    compiler: string;
    /** One of 0 1 2 3 S G F; F is -Ofast. */
    optim: string;
    /** "c++17" (v4 and up) or "17" (earlier). */
    cppVersion: string;
    /** "gnu" for libstdc++ or "llvm" for libc++; clang only. Absent before protocol 3. */
    lib?: string;
    /** Extra flags, from a per-compiler allow-list. Absent before protocol 5. */
    flags?: string[];
};

/** Google Benchmark 1.6.1 is the version every Quick Bench image was built with. */
export const QUICK_BENCH_BENCHMARK_LIB = {name: 'benchmark', ver: '161'};
/** Quick Bench images had range-v3 0.3.0 in /usr/include, so code could use it without asking. */
export const QUICK_BENCH_RANGES_LIB = {name: 'rangesv3', ver: '030'};

export type QuickBenchConversion = {
    state: ClientState;
    /** Anything the conversion had to guess or drop; worth logging, and maybe worth showing the user. */
    warnings: string[];
};

const OPTIM_FLAGS: Readonly<Record<string, string>> = {
    '0': '-O0',
    '1': '-O1',
    '2': '-O2',
    '3': '-O3',
    S: '-Os',
    G: '-Og',
    F: '-Ofast',
};

const CPP_VERSION_RE = /^(?:(?:c|gnu)\+\+)?(\w+)$/;
const RANGES_V3_RE = /#\s*include\s*[<"]range\/v3\//;

function stdFlag(cppVersion: string, warnings: string[]): string | undefined {
    const match = CPP_VERSION_RE.exec(cppVersion);
    if (!match) {
        warnings.push(`Ignored unrecognised C++ version "${cppVersion}"`);
        return undefined;
    }
    // Quick Bench prefixed bare versions with c++, never gnu++.
    return cppVersion.includes('++') ? `-std=${cppVersion}` : `-std=c++${match[1]}`;
}

/** Builds the compiler options string Quick Bench would have passed, as far as CE's compiler needs it. */
export function quickBenchCompilerOptions(options: QuickBenchOptions, warnings: string[] = []): string {
    const result: string[] = [];
    const std = stdFlag(String(options.cppVersion), warnings);
    if (std) result.push(std);

    const optim = OPTIM_FLAGS[String(options.optim)];
    if (optim) result.push(optim);
    else warnings.push(`Ignored unrecognised optimisation level "${options.optim}"`);

    if (options.lib === 'llvm') {
        // Quick Bench disabled the library choice for gcc, so libc++ only ever went with clang.
        if (options.compiler.startsWith('clang')) result.push('-stdlib=libc++');
        else warnings.push(`Ignored libc++ for non-clang compiler ${options.compiler}`);
    }

    for (const flag of options.flags ?? []) {
        if (typeof flag === 'string' && /^-\S+$/.test(flag)) result.push(flag);
        else warnings.push(`Ignored malformed flag ${JSON.stringify(flag)}`);
    }
    return result.join(' ');
}

function makeSession(id: number, source: string, options: QuickBenchOptions, candidates: QuickBenchCandidates) {
    const warnings: string[] = [];
    const resolution = resolveQuickBenchCompiler(options.compiler, candidates);
    if (resolution.note) warnings.push(resolution.note);
    const libs = [QUICK_BENCH_BENCHMARK_LIB];
    if (RANGES_V3_RE.test(source)) libs.push(QUICK_BENCH_RANGES_LIB);
    const session = {
        id,
        language: 'c++',
        source,
        compilers: [],
        executors: [
            {
                compiler: {
                    id: resolution.id ?? '',
                    options: quickBenchCompilerOptions(options, warnings),
                    libs: libs.map(lib => ({...lib})),
                    filters: {},
                },
                compilerVisible: true,
                compilerOutputVisible: true,
                arguments: '',
                stdin: '',
            },
        ],
    };
    return {session, warnings};
}

function parseStoredOptions(optJson: string): QuickBenchOptions {
    const parsed = JSON.parse(optJson);
    if (typeof parsed !== 'object' || parsed === null || typeof parsed.compiler !== 'string') {
        throw new Error('Quick Bench options have no compiler');
    }
    return {
        compiler: parsed.compiler,
        optim: String(parsed.optim ?? '3'),
        cppVersion: String(parsed.cppVersion ?? '17'),
        lib: parsed.lib,
        flags: Array.isArray(parsed.flags) ? parsed.flags : [],
    };
}

/**
 * Converts a stored Quick Bench benchmark (the contents of its .cpp and .opt files, any protocol from 1 to 5) to CE
 * client state.
 */
export function convertStoredQuickBench(
    storedCode: string,
    optJson: string,
    candidates: QuickBenchCandidates,
): QuickBenchConversion {
    const options = parseStoredOptions(optJson);
    const source = wrapQuickBenchCode(unwrapQuickBenchCode(storedCode));
    const {session, warnings} = makeSession(1, source, options, candidates);
    return {state: new ClientState({sessions: [session]}), warnings};
}

/** What other sites (CE among them) put in a quick-bench.com/#... link. All but `text` are optional. */
type QuickBenchHashEntry = {
    text?: unknown;
    compiler?: unknown;
    optim?: unknown;
    cppVersion?: unknown;
    lib?: unknown;
    flags?: unknown;
};

/** Quick Bench's front-end defaults for anything a #fragment link left out. */
const HASH_DEFAULTS = {compiler: QUICK_BENCH_DEFAULT_COMPILER, optim: '3', cppVersion: '20', lib: 'gnu'};

function decodeHashPayload(fragment: string): unknown {
    let payload = fragment.startsWith('#') ? fragment.slice(1) : fragment;
    if (payload.includes('%')) payload = decodeURIComponent(payload);
    // Senders base64 either UTF-8 bytes or JSON with non-ASCII escaped; decoding as UTF-8 handles both.
    return JSON.parse(Buffer.from(payload, 'base64').toString('utf8'));
}

function stringOr(value: unknown, fallback: string): string {
    return typeof value === 'string' || typeof value === 'number' ? String(value) : fallback;
}

/**
 * Converts the state in a quick-bench.com/#<base64 JSON> link to CE client state, one session per benchmark in it.
 * Returns undefined if the fragment holds nothing Quick Bench would have opened.
 */
export function convertQuickBenchHash(
    fragment: string,
    candidates: QuickBenchCandidates,
): QuickBenchConversion | undefined {
    let decoded: unknown;
    try {
        decoded = decodeHashPayload(fragment);
    } catch {
        return undefined;
    }
    const entries = ([] as unknown[])
        .concat(decoded)
        .filter((e): e is QuickBenchHashEntry => typeof e === 'object' && e !== null)
        .filter(e => typeof e.text === 'string' && e.text !== '');
    if (entries.length === 0) return undefined;

    const sessions: ReturnType<typeof makeSession>['session'][] = [];
    const warnings: string[] = [];
    for (const [index, entry] of entries.entries()) {
        const options: QuickBenchOptions = {
            compiler: stringOr(entry.compiler, HASH_DEFAULTS.compiler),
            optim: stringOr(entry.optim, HASH_DEFAULTS.optim),
            cppVersion: stringOr(entry.cppVersion, HASH_DEFAULTS.cppVersion),
            lib: stringOr(entry.lib, HASH_DEFAULTS.lib),
            flags: Array.isArray(entry.flags) ? entry.flags : [],
        };
        const source = wrapQuickBenchCode(stripQuickBenchHashBoilerplate(entry.text as string));
        const made = makeSession(index + 1, source, options, candidates);
        sessions.push(made.session);
        warnings.push(...made.warnings);
    }
    return {state: new ClientState({sessions}), warnings};
}
