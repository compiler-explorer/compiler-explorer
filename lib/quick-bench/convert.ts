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
    /** Extra flags. Absent before protocol 5. */
    flags?: string[];
};

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

// What Quick Bench accepted: an optional "c++" and a two-character version such as 17 or 2a.
const CPP_VERSION_RE = /^(?:c\+\+)?(\d[\da-z])$/;
const RANGES_V3_RE = /#\s*include\s*[<"]range\/v3\//;

/**
 * Builds the compiler options for a Quick Bench benchmark.
 *
 * This is what the user chose, which isn't always what Quick Bench ran: from protocol 5 it ran the lib choice and
 * the flags together as one argument, so a benchmark with flags actually got libstdc++ and lost its first flag.
 */
export function quickBenchCompilerOptions(options: QuickBenchOptions, warnings: string[] = []): string {
    const result: string[] = [];
    const std = CPP_VERSION_RE.exec(options.cppVersion);
    if (std) result.push(`-std=c++${std[1]}`);
    else warnings.push(`Ignored unrecognised C++ version "${options.cppVersion}"`);

    const optim = OPTIM_FLAGS[options.optim];
    if (optim) result.push(optim);
    else warnings.push(`Ignored unrecognised optimisation level "${options.optim}"`);

    if (options.lib === 'llvm') {
        // Quick Bench disabled the library choice for gcc, so libc++ only ever went with clang.
        if (options.compiler.startsWith('clang')) result.push('-stdlib=libc++');
        else warnings.push(`Ignored libc++ for non-clang compiler ${options.compiler}`);
    }

    // Whitespace would split one flag into several options.
    for (const flag of options.flags ?? []) {
        if (typeof flag === 'string' && /^-\S+$/.test(flag)) result.push(flag);
        else warnings.push(`Ignored malformed flag ${JSON.stringify(flag)}`);
    }
    return result.join(' ');
}

function convert(code: string, options: QuickBenchOptions, candidates: QuickBenchCandidates): QuickBenchConversion {
    const warnings: string[] = [];
    const resolution = resolveQuickBenchCompiler(options.compiler, candidates);
    if (resolution.note) warnings.push(resolution.note);
    const source = wrapQuickBenchCode(code);
    // Google Benchmark 1.6.1 is what every Quick Bench image was built with, and they all had range-v3 0.3.0 in
    // /usr/include, so code could use it without asking.
    const libs = [{name: 'benchmark', ver: '161'}];
    if (RANGES_V3_RE.test(source)) libs.push({name: 'rangesv3', ver: '030'});
    const session = {
        id: 1,
        language: 'c++',
        source,
        executors: [
            {
                compiler: {id: resolution.id ?? '', options: quickBenchCompilerOptions(options, warnings), libs},
                compilerVisible: true,
                compilerOutputVisible: true,
                arguments: '',
            },
        ],
    };
    return {state: new ClientState({sessions: [session]}), warnings};
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
    const parsed = JSON.parse(optJson);
    if (typeof parsed?.compiler !== 'string') throw new Error('Quick Bench options have no compiler');
    const options = {
        compiler: parsed.compiler,
        optim: String(parsed.optim ?? '3'),
        cppVersion: String(parsed.cppVersion ?? '17'),
        lib: parsed.lib,
        flags: Array.isArray(parsed.flags) ? parsed.flags : [],
    };
    return convert(unwrapQuickBenchCode(storedCode), options, candidates);
}

/** Quick Bench's front-end defaults for anything a #fragment link left out. */
const HASH_DEFAULTS = {compiler: QUICK_BENCH_DEFAULT_COMPILER, optim: '3', cppVersion: '20', lib: 'gnu'};

function stringOr(value: unknown, fallback: string): string {
    return typeof value === 'string' || typeof value === 'number' ? String(value) : fallback;
}

/**
 * Converts the state in a quick-bench.com/#<base64 JSON> link to CE client state. The JSON is an object or (from
 * Build Bench, which shares the format) an array of them; like Quick Bench, this takes the first. Returns undefined
 * if the fragment holds nothing Quick Bench would have opened.
 */
export function convertQuickBenchHash(
    fragment: string,
    candidates: QuickBenchCandidates,
): QuickBenchConversion | undefined {
    let decoded: unknown;
    try {
        let payload = fragment.startsWith('#') ? fragment.slice(1) : fragment;
        if (payload.includes('%')) payload = decodeURIComponent(payload);
        // Senders base64 either UTF-8 bytes or JSON with non-ASCII escaped; decoding as UTF-8 handles both.
        decoded = JSON.parse(Buffer.from(payload, 'base64').toString('utf8'));
    } catch {
        return undefined;
    }
    const entry = ([] as any[]).concat(decoded).find(e => typeof e?.text === 'string' && e.text !== '');
    if (!entry) return undefined;
    const options = {
        compiler: stringOr(entry.compiler, HASH_DEFAULTS.compiler),
        optim: stringOr(entry.optim, HASH_DEFAULTS.optim),
        cppVersion: stringOr(entry.cppVersion, HASH_DEFAULTS.cppVersion),
        lib: stringOr(entry.lib, HASH_DEFAULTS.lib),
        flags: Array.isArray(entry.flags) ? entry.flags : [],
    };
    return convert(stripQuickBenchHashBoilerplate(entry.text), options, candidates);
}
