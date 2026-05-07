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

import {inferReleaseTrack, isReleaseTrack} from '../lib/release-track.js';

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

        it('"(snapshot)" tag (powerpc clang and similar)', () => {
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(snapshot)'})).toBe('nightly');
        });

        it('parenthesised nightly tags', () => {
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(main)'})).toBe('nightly');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(master)'})).toBe('nightly');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: '(nightly)'})).toBe('nightly');
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
        it('1.x.y-preview (micropython, dxc)', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: true, semver: '1.28.0-preview'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '1.8.2306-preview'})).toBe(
                'prerelease',
            );
        });

        it('semver -rc / -alpha / -beta segments', () => {
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '1.0.0-rc1'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '2.0.0-alpha.1'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: true, isNightly: false, semver: '3.0.0-beta'})).toBe('prerelease');
        });

        it('prerelease tags with whitespace and mixed case', () => {
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: ' BETA '})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'RC1'})).toBe('prerelease');
            expect(inferReleaseTrack({isSemVer: false, isNightly: true, semver: 'Alpha'})).toBe('prerelease');
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
