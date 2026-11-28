// Copyright (c) 2026, Compiler Explorer Authors
// Portions derived from Quick Bench (https://github.com/FredTingaud/quick-bench-back-end,
// https://github.com/FredTingaud/quick-bench-front-end, https://github.com/FredTingaud/quick-bench-docker),
// Copyright (c) 2017, Fred Tingaud, used under the BSD 2-Clause License.
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

import type {CompilerInfo} from '../../types/compiler.interfaces.js';

// Quick Bench named a compiler by family and major.minor ("clang-11.1", "gcc-13.2"; "clang++-3.8" in its first
// protocols), and each name was a Docker image built from one exact release. The name alone doesn't give the release
// (clang-11.1 was 11.1.0 but clang-11.0 was 11.0.1), so we look up the image's release and find that among CE's
// compilers, falling back to the nearest version when CE lacks it.

// From quick-bench-docker's containers.json `release` tags, except: clang-3.8 and clang-3.9 come from their
// Dockerfiles, and clang-17.0 was only ever published to Docker Hub (its image sets CLANG_RELEASE=llvmorg-17.0.1).
// The "clang++-3.8" of protocols 1 and 2 predates the Docker images; we can't know its patch level, so it's taken to
// be the same as the clang-3.8 image.
/** The exact clang release behind each Quick Bench clang image. */
export const QUICK_BENCH_CLANG_RELEASES: Readonly<Record<string, string>> = {
    '3.8': '3.8.1',
    '3.9': '3.9.1',
    '4.0': '4.0.1',
    '5.0': '5.0.2',
    '6.0': '6.0.1',
    '7.0': '7.0.1',
    '7.1': '7.1.0',
    '8.0': '8.0.1',
    '9.0': '9.0.1',
    '10.0': '10.0.1',
    '11.0': '11.0.1',
    '11.1': '11.1.0',
    '12.0': '12.0.1',
    '13.0': '13.0.1',
    '14.0': '14.0.6',
    '15.0': '15.0.7',
    '16.0': '16.0.0',
    '17.0': '17.0.1',
};

/** Quick Bench's gcc images, from containers.json. Each was built from gcc-<major>.<minor>.0. */
export const QUICK_BENCH_GCC_VERSIONS: ReadonlySet<string> = new Set([
    '5.5',
    '6.4',
    '6.5',
    '7.2',
    '7.3',
    '7.4',
    '7.5',
    '8.1',
    '8.2',
    '8.3',
    '8.4',
    '8.5',
    '9.1',
    '9.2',
    '9.3',
    '9.4',
    '9.5',
    '10.1',
    '10.2',
    '10.3',
    '10.4',
    '10.5',
    '11.1',
    '11.2',
    '11.3',
    '11.4',
    '12.1',
    '12.2',
    '12.3',
    '13.1',
    '13.2',
]);

/** The compiler Quick Bench's front end selected when a #fragment link named none: its newest. */
export const QUICK_BENCH_DEFAULT_COMPILER = 'clang-17.0';

type QuickBenchCompilerFamily = 'clang' | 'gcc';

export type QuickBenchCompiler = {
    family: QuickBenchCompilerFamily;
    /** The exact release Quick Bench ran, as [major, minor, patch]. */
    version: [number, number, number];
    /** False when the name isn't one of Quick Bench's images, and the version is a guess from the name. */
    known: boolean;
};

const QUICK_BENCH_COMPILER_RE = /^(clang|gcc)(?:\+\+)?-(\d+)\.(\d+)$/;

export function parseQuickBenchCompiler(name: string): QuickBenchCompiler | undefined {
    const match = QUICK_BENCH_COMPILER_RE.exec(name);
    if (!match) return undefined;
    const family = match[1] as QuickBenchCompilerFamily;
    const majorMinor = `${Number(match[2])}.${Number(match[3])}`;
    if (family === 'clang') {
        const release = QUICK_BENCH_CLANG_RELEASES[majorMinor];
        if (release) return {family, version: parseVersion(release) as [number, number, number], known: true};
    }
    const known = family === 'gcc' && QUICK_BENCH_GCC_VERSIONS.has(majorMinor);
    return {family, version: [Number(match[2]), Number(match[3]), 0], known};
}

/** Parses a plain dotted version of one to three numbers, padding to three. Anything else is not a release. */
export function parseVersion(semver: string): [number, number, number] | undefined {
    const match = /^(\d+)(?:\.(\d+))?(?:\.(\d+))?$/.exec(semver.trim());
    if (!match) return undefined;
    return [Number(match[1]), Number(match[2] ?? 0), Number(match[3] ?? 0)];
}

export type CeCompilerCandidate = {id: string; semver: string};

/** The CE compilers a Quick Bench compiler may map to, by family. */
export type QuickBenchCandidates = Record<QuickBenchCompilerFamily, CeCompilerCandidate[]>;

export type CompilerResolution = {
    /** The CE compiler id, or undefined when CE has no compiler of this family at all. */
    id: string | undefined;
    /** True when CE has the very release Quick Bench ran. */
    exact: boolean;
    /** Why the mapping is not exact, for logging and for showing the user. */
    note?: string;
};

function formatVersion(version: readonly number[]) {
    return version.join('.');
}

function versionKey(version: readonly number[]): number {
    return (version[0] * 1000 + version[1]) * 1000 + version[2];
}

// Closest major first, then closest release within it: so 11.1.0's nearest is 11.0.1 rather than 11.0.0.
function versionDistance(a: readonly number[], b: readonly number[]): number[] {
    return [Math.abs(a[0] - b[0]), Math.abs(versionKey(a) - versionKey(b))];
}

function compareTuples(a: number[], b: number[]): number {
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return a[i] - b[i];
    }
    return 0;
}

/**
 * Maps a Quick Bench compiler name to a CE compiler id: the CE compiler of the same family with the very release
 * Quick Bench ran, else the nearest one (closest major, then closest release within it; the newer on a tie).
 */
export function resolveQuickBenchCompiler(name: string, candidates: QuickBenchCandidates): CompilerResolution {
    const qb = parseQuickBenchCompiler(name);
    if (!qb) return {id: undefined, exact: false, note: `Unknown Quick Bench compiler "${name}"`};

    let best: {id: string; version: number[]; distance: number[]} | undefined;
    for (const candidate of candidates[qb.family]) {
        const version = parseVersion(candidate.semver);
        if (!version) continue;
        const distance = versionDistance(version, qb.version);
        const order = best ? compareTuples(distance, best.distance) : -1;
        if (order < 0 || (order === 0 && best && compareTuples(version, best.version) > 0)) {
            best = {id: candidate.id, version, distance};
        }
    }

    const wanted = `${qb.family} ${formatVersion(qb.version)}`;
    if (!best) return {id: undefined, exact: false, note: `No ${qb.family} compiler available for ${name} (${wanted})`};
    const exact = best.distance.every(d => d === 0);
    if (exact) return {id: best.id, exact: true};
    const guess = qb.known ? '' : `, a guess from the name "${name}"`;
    return {
        id: best.id,
        exact: false,
        note: `Quick Bench's ${name} was ${wanted}${guess}; nearest available is ${best.id} (${formatVersion(best.version)})`,
    };
}

// The groups of plain x86-64 gcc and clang releases in godbolt.org's configuration (`c++.amazon.properties`).
const CANDIDATE_GROUPS: Readonly<Record<QuickBenchCompilerFamily, string>> = {gcc: 'gcc86', clang: 'clang'};

/** Picks the CE compilers Quick Bench compilers may map to. */
export function candidatesFromCompilers(
    compilers: readonly Pick<CompilerInfo, 'id' | 'group' | 'semver' | 'lang'>[],
): QuickBenchCandidates {
    const pick = (family: QuickBenchCompilerFamily) =>
        compilers
            .filter(c => c.lang === 'c++' && c.group === CANDIDATE_GROUPS[family])
            .map(c => ({id: c.id, semver: c.semver}));
    return {gcc: pick('gcc'), clang: pick('clang')};
}
