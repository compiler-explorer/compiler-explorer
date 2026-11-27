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

import fs from 'node:fs';

import {describe, expect, it} from 'vitest';

import {parseProperties} from '../../lib/properties.js';
import {
    candidatesFromCompilers,
    parseQuickBenchCompiler,
    parseVersion,
    QUICK_BENCH_CLANG_RELEASES,
    QUICK_BENCH_GCC_VERSIONS,
    type QuickBenchCandidates,
    resolveQuickBenchCompiler,
} from '../../lib/quick-bench/compilers.js';
import type {CompilerInfo} from '../../types/compiler.interfaces.js';

describe('Parsing Quick Bench compiler names', () => {
    it.each([
        ['clang-11.1', {family: 'clang', version: [11, 1, 0], known: true}],
        ['clang-11.0', {family: 'clang', version: [11, 0, 1], known: true}],
        ['clang++-3.8', {family: 'clang', version: [3, 8, 1], known: true}],
        ['clang-17.0', {family: 'clang', version: [17, 0, 1], known: true}],
        ['gcc-5.5', {family: 'gcc', version: [5, 5, 0], known: true}],
        ['clang-9.1', {family: 'clang', version: [9, 1, 0], known: false}],
        ['gcc-14.1', {family: 'gcc', version: [14, 1, 0], known: false}],
    ])('parses %s', (name, expected) => {
        expect(parseQuickBenchCompiler(name)).toEqual(expected);
    });

    it.each(['', 'icc-19.0', 'clang', 'clang-17', 'gcc-13.2.0', 'clang-17.0 -fplugin'])('rejects %j', name => {
        expect(parseQuickBenchCompiler(name)).toBeUndefined();
    });
});

describe('Resolving Quick Bench compilers', () => {
    const candidates: QuickBenchCandidates = {
        gcc: [
            {id: 'g55', semver: 'not a version'},
            {id: 'g550', semver: '5.5'},
            {id: 'g125', semver: '12.5'},
        ],
        clang: [
            {id: 'clang1100', semver: '11.0.0'},
            {id: 'clang1101', semver: '11.0.1'},
            {id: 'clang1400', semver: '14.0.0'},
            {id: 'clang1500', semver: '15.0.0'},
        ],
    };

    it('finds the exact release', () => {
        expect(resolveQuickBenchCompiler('gcc-5.5', candidates)).toEqual({id: 'g550', exact: true});
        expect(resolveQuickBenchCompiler('clang-11.0', candidates)).toEqual({id: 'clang1101', exact: true});
    });

    it('falls back to the nearest release in the same major, and says so', () => {
        const resolved = resolveQuickBenchCompiler('clang-11.1', candidates);
        expect(resolved.id).toEqual('clang1101');
        expect(resolved.exact).toBe(false);
        expect(resolved.note).toContain('11.1.0');
        expect(resolveQuickBenchCompiler('clang-14.0', candidates).id).toEqual('clang1400');
        expect(resolveQuickBenchCompiler('gcc-12.3', candidates).id).toEqual('g125');
    });

    it('prefers a closer major over a closer number', () => {
        expect(resolveQuickBenchCompiler('clang-13.0', candidates).id).toEqual('clang1400');
        expect(resolveQuickBenchCompiler('clang-12.0', candidates).id).toEqual('clang1101');
    });

    it('marks names Quick Bench never had as guesses', () => {
        expect(resolveQuickBenchCompiler('clang-11.2', candidates).note).toContain('guess');
    });

    it('returns no id when it cannot', () => {
        expect(resolveQuickBenchCompiler('gcc-5.5', {gcc: [], clang: []}).id).toBeUndefined();
        expect(resolveQuickBenchCompiler('msvc-19.0', candidates).id).toBeUndefined();
    });
});

describe('Picking candidates', () => {
    it('takes C++ compilers from the configured groups', () => {
        const compilers: Pick<CompilerInfo, 'id' | 'group' | 'semver' | 'lang'>[] = [
            {id: 'g132', group: 'gcc86', semver: '13.2', lang: 'c++'},
            {id: 'g132assert', group: 'gcc86assert', semver: '13.2 (assertions)', lang: 'c++'},
            {id: 'cg132', group: 'gcc86', semver: '13.2', lang: 'c'},
            {id: 'clang1701', group: 'clang', semver: '17.0.1', lang: 'c++'},
            {id: 'armv8-clang1701', group: 'armv8-clang', semver: '17.0.1', lang: 'c++'},
        ];
        expect(candidatesFromCompilers(compilers)).toEqual({
            gcc: [{id: 'g132', semver: '13.2'}],
            clang: [{id: 'clang1701', semver: '17.0.1'}],
        });
    });
});

describe("Quick Bench's compilers on godbolt.org", () => {
    const props = parseProperties(fs.readFileSync('etc/config/c++.amazon.properties', 'utf8'), 'c++.amazon');
    const groupCandidates = (group: string) =>
        String(props[`group.${group}.compilers`])
            .split(':')
            .filter(id => !id.startsWith('&'))
            .map(id => ({id, semver: String(props[`compiler.${id}.semver`] ?? '')}));
    const candidates: QuickBenchCandidates = {gcc: groupCandidates('gcc86'), clang: groupCandidates('clang')};

    const images = [
        ...Object.keys(QUICK_BENCH_CLANG_RELEASES).map(v => `clang-${v}`),
        ...[...QUICK_BENCH_GCC_VERSIONS].map(v => `gcc-${v}`),
    ];

    it.each(images)('maps %s to a release of the same major', name => {
        const resolved = resolveQuickBenchCompiler(name, candidates);
        expect(resolved.id).toBeDefined();
        const semver = candidates[name.startsWith('gcc') ? 'gcc' : 'clang'].find(c => c.id === resolved.id)?.semver;
        const [major] = parseVersion(semver ?? '') ?? [];
        expect(major).toEqual(parseQuickBenchCompiler(name)?.version[0]);
    });
});
