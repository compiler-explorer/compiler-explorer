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

import {describe, expect, it} from 'vitest';

import {backfillReleaseTrack, inferReleaseTrack, isReleaseTrack, resolveReleaseTrack} from '../lib/release-track.js';

describe('inferReleaseTrack', () => {
    describe('stable', () => {
        it('numbered semver release', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '14.2.0'})).toBe('stable');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '1.95.0'})).toBe('stable');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '20.1'})).toBe('stable');
        });

        it('falls back to stable for unknown isSemVer=false non-nightly compilers', () => {
            // E.g. a one-off compiler with no version metadata.
            expect(inferReleaseTrack({isSemVer: false, isNightly: false, semver: ''})).toBe('stable');
            expect(inferReleaseTrack({isSemVer: false, isNightly: false, semver: 'unknown'})).toBe('stable');
        });
    });

    describe('nightly', () => {
        it('semver containing "trunk" (gcc snapshot, clang trunk)', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '(trunk)'})).toBe('nightly');
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: 'trunk'})).toBe('nightly');
        });

        it('semver containing "main"', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: 'main'})).toBe('nightly');
        });

        it('bare "nightly" tag (rust nightly)', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: 'nightly'})).toBe('nightly');
        });

        it('case insensitive', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: 'NIGHTLY'})).toBe('nightly');
        });

        it('parenthesised nightly tags', () => {
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(main)'})).toBe('nightly');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(master)'})).toBe('nightly');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(nightly)'})).toBe('nightly');
        });

        it('"(tip)" tag (Go gltip variants — Go calls their mainline "tip")', () => {
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(tip)'})).toBe('nightly');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'tip'})).toBe('nightly');
        });

        it('isNightly with empty semver — the canonical "just a nightly" pattern', () => {
            // Catches the common case where a maintainer flags isNightly=true but doesn't
            // bother to write a semver at all (wasm32clang, flangtrunk, dotnettrunk*,
            // rustccggcc-master, etc.) — i.e. "nothing fancy, just the trunk build".
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: ''})).toBe('nightly');
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: ''})).toBe('nightly');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '   '})).toBe('nightly');
        });
    });

    describe('prerelease', () => {
        it('beta tag (rust beta)', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: 'beta'})).toBe('prerelease');
        });

        it('alpha tag', () => {
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'alpha'})).toBe('prerelease');
        });

        it('rc / rc1 / rc2 tags', () => {
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'rc'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'rc1'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'rc12'})).toBe('prerelease');
        });
    });

    describe('prerelease — real semver with prerelease segment', () => {
        it('1.x.y-preview without isNightly (dxc-style RC of an upcoming version)', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '1.8.2306-preview'})).toBe(
                'prerelease',
            );
        });

        it('isNightly + prerelease segment promotes to nightly (micropython-preview is a rolling preview)', () => {
            // The maintainer's explicit isNightly=true overrides the generic prerelease
            // interpretation: -preview here means "rolling preview branch", not "RC".
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '1.28.0-preview'})).toBe('nightly');
        });

        it('semver -rc / -alpha / -beta segments', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '1.0.0-rc1'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '2.0.0-alpha.1'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '3.0.0-beta'})).toBe('prerelease');
        });

        it('semver -dev / -pre segments', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '5.0.0-dev'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '5.0.0-pre1'})).toBe('prerelease');
        });

        it('prerelease tags with whitespace and mixed case', () => {
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: ' BETA '})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'RC1'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'Alpha'})).toBe('prerelease');
        });
    });

    describe('stable — real semver with unrecognised suffix is NOT a prerelease', () => {
        it('build flavour suffix (OCaml flambda)', () => {
            // ocaml4071flambda etc. — "-flambda" is a build flavour, not a release-track signal.
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '4.07.1-flambda'})).toBe('stable');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '4.14.0-flambda'})).toBe('stable');
        });

        it('distro revision suffix (Alire-packaged GNAT)', () => {
            // gnatarm103, gnatriscv64112 etc. — "-2" is the distro revision number.
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '10.3.0-2'})).toBe('stable');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '11.2.0-3'})).toBe('stable');
        });

        it('git hash suffix (cmake snapshots)', () => {
            // cmake-3_29_20240506_g1ea8fa8 — "-g1ea8fa8" is a git short hash.
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '3.29.20240506-g1ea8fa8'})).toBe(
                'stable',
            );
        });

        it('vendor codename suffix (KVX ACB)', () => {
            // kvxg750/ckvxg750 — "-cd1" is a vendor codename.
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '4.1.0-cd1'})).toBe('stable');
        });
    });

    describe('"snapshot" is deliberately not a nightly tag', () => {
        it('semver=(snapshot) without isNightly stays stable', () => {
            // IBM Advance Toolchain pattern: ppc64g8/ppc64leg8 etc. are numbered AT
            // releases (AT12.0, AT13.0) that happen to be built from upstream gcc
            // snapshots. The (snapshot) tag is descriptive, not a nightly indicator.
            expect(inferReleaseTrack({isSemVer: false, isNightly: false, semver: '(snapshot)'})).toBe('stable');
        });

        it('semver=(snapshot) with isNightly classifies as experimental — needs explicit override', () => {
            // ppc64clang etc. are genuine clang trunk builds. The override
            // mechanism (compiler.releaseTrack=nightly in .properties) handles them.
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(snapshot)'})).toBe('experimental');
        });
    });

    describe('experimental', () => {
        it('isNightly with non-canonical semver tag (gcc contracts/modules forks)', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '(contracts)'})).toBe('experimental');
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '(modules)'})).toBe('experimental');
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '(coroutines)'})).toBe('experimental');
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '(P2034 lambdas)'})).toBe(
                'experimental',
            );
        });
    });

    describe('precedence', () => {
        it('prefers stable over experimental when isSemVer parses cleanly even with isNightly set', () => {
            // Defensive: shouldn't happen in real configs but the rules need to be deterministic.
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '14.2.0'})).toBe('stable');
        });

        it('prefers nightly over experimental for canonical trunk despite isNightly', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '(trunk)'})).toBe('nightly');
        });

        it('prefers prerelease over experimental for known prerelease tags', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: 'beta'})).toBe('prerelease');
        });
    });
});

describe('isReleaseTrack', () => {
    it('accepts the four known tracks', () => {
        expect(isReleaseTrack('stable')).toBe(true);
        expect(isReleaseTrack('nightly')).toBe(true);
        expect(isReleaseTrack('prerelease')).toBe(true);
        expect(isReleaseTrack('experimental')).toBe(true);
    });

    it('rejects anything else', () => {
        expect(isReleaseTrack('')).toBe(false);
        expect(isReleaseTrack('STABLE')).toBe(false);
        expect(isReleaseTrack('stab')).toBe(false);
        expect(isReleaseTrack('release')).toBe(false);
    });
});

describe('resolveReleaseTrack', () => {
    const stableInputs = {isSemVer: true, isNightly: false, semver: '14.2.0'};

    it('falls back to inference when override is empty', () => {
        expect(resolveReleaseTrack('', stableInputs, 'gcc-test')).toBe('stable');
    });

    it('falls back to inference when override is whitespace only', () => {
        expect(resolveReleaseTrack('   ', stableInputs, 'gcc-test')).toBe('stable');
        expect(resolveReleaseTrack('\t\n', stableInputs, 'gcc-test')).toBe('stable');
    });

    it('a valid override beats the inferred value', () => {
        // Inputs would infer 'stable' but override forces 'nightly'.
        expect(resolveReleaseTrack('nightly', stableInputs, 'gcc-test')).toBe('nightly');
        expect(resolveReleaseTrack('experimental', stableInputs, 'gcc-test')).toBe('experimental');
    });

    it('trims whitespace around a valid override', () => {
        expect(resolveReleaseTrack('  nightly  ', stableInputs, 'gcc-test')).toBe('nightly');
    });

    it('rejects an unknown override string with a clear message', () => {
        expect(() => resolveReleaseTrack('stabel', stableInputs, 'gcc-test')).toThrow(/stabel.*gcc-test/);
        expect(() => resolveReleaseTrack('release', stableInputs, 'gcc-test')).toThrow(/gcc-test/);
    });

    it('rejects uppercase override values (case sensitive)', () => {
        expect(() => resolveReleaseTrack('NIGHTLY', stableInputs, 'gcc-test')).toThrow(/NIGHTLY.*gcc-test/);
    });

    it('rejects non-string raw values (toProperty coerces "true"/"1" to bool/number)', () => {
        // toProperty in lib/utils.ts coerces truthy/numeric strings; resolveReleaseTrack
        // must reject these up-front rather than crashing on .trim().
        expect(() => resolveReleaseTrack(true as unknown as string, stableInputs, 'gcc-test')).toThrow(
            /expected a string/,
        );
        expect(() => resolveReleaseTrack(1 as unknown as string, stableInputs, 'gcc-test')).toThrow(
            /expected a string/,
        );
        expect(() => resolveReleaseTrack(null as unknown as string, stableInputs, 'gcc-test')).toThrow(
            /expected a string/,
        );
    });
});

describe('backfillReleaseTrack', () => {
    it('keeps a valid existing releaseTrack', () => {
        const c = {isSemVer: true, isNightly: false, semver: '14.2.0', releaseTrack: 'experimental' as const};
        backfillReleaseTrack(c);
        expect(c.releaseTrack).toBe('experimental');
    });

    it('infers when releaseTrack is missing', () => {
        const c: any = {isSemVer: true, isNightly: false, semver: '14.2.0'};
        backfillReleaseTrack(c);
        expect(c.releaseTrack).toBe('stable');
    });

    it('re-infers when releaseTrack is invalid (defends against hand-edited JSON)', () => {
        const c: any = {isSemVer: true, isNightly: true, semver: 'nightly', releaseTrack: 'stabel'};
        backfillReleaseTrack(c);
        expect(c.releaseTrack).toBe('nightly');
    });

    it('handles undefined releaseTrack the same as missing', () => {
        const c: any = {isSemVer: true, isNightly: true, semver: '', releaseTrack: undefined};
        backfillReleaseTrack(c);
        expect(c.releaseTrack).toBe('nightly');
    });
});
