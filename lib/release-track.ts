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

import semverParser from 'semver';

import {RELEASE_TRACKS, type ReleaseTrack} from '../types/compiler.interfaces.js';
import {asSafeVer, magic_semver} from './utils.js';

const PRERELEASE_TAGS = new Set(['beta', 'alpha']);
const NIGHTLY_TAGS = new Set(['nightly', 'main', 'master', 'snapshot']);
const RC_PATTERN = /^rc\d*$/;

// CE configs commonly write nightly tags inside parens, e.g. "(trunk)", "(snapshot)",
// "(main)". Strip an outer paren wrapper so the tag-set membership tests can match
// regardless of whether the maintainer wrote `semver=trunk` or `semver=(trunk)`.
function stripParens(s: string): string {
    const m = s.match(/^\((.*)\)$/);
    return m ? m[1].trim() : s;
}

export type ReleaseTrackInputs = {
    isSemVer: boolean;
    isNightly: boolean;
    semver: string;
};

/**
 * Categorise a compiler by release track based on the metadata CE already collects.
 *
 * The decision tree (first match wins):
 *   1. Real numbered semver with a prerelease segment (e.g. "1.28.0-preview") →
 *      'prerelease'. Caught before rule 2 so genuine prereleases aren't mislabelled
 *      as 'stable'.
 *   2. Real numbered semver, no prerelease segment → 'stable'.
 *   3. asSafeVer maps to magic_semver.trunk (semver contains "trunk"/"main"), or the
 *      bare semver tag is "nightly" / "main" / "master" → 'nightly'.
 *   4. semver is a prerelease tag ("beta", "alpha", "rc", "rc1", ...) → 'prerelease'.
 *   5. isNightly with any other non-numeric semver (e.g. "(contracts)", "(modules)")
 *      → 'experimental' — branch builds for testing language proposals.
 *   6. Anything else → 'stable' as the safe fallback.
 *
 * Cases the heuristic can't reach from structural fields alone (e.g. Rust's
 * `rustccggcc-master` / `mrustc-master`, where "master" lives in the compiler id
 * but not the semver) should set `compiler.releaseTrack=nightly` in the .properties
 * file — that's what the override is for.
 */
export function inferReleaseTrack(inputs: ReleaseTrackInputs): ReleaseTrack {
    const semver = inputs.semver.toLowerCase().trim();
    const tag = stripParens(semver);
    const safe = asSafeVer(semver);
    const isMagic = safe === magic_semver.trunk || safe === magic_semver.non_trunk;

    if (inputs.isSemVer && !isMagic) {
        return semverParser.prerelease(safe) ? 'prerelease' : 'stable';
    }
    if (safe === magic_semver.trunk || NIGHTLY_TAGS.has(tag)) return 'nightly';
    if (PRERELEASE_TAGS.has(tag) || RC_PATTERN.test(tag)) return 'prerelease';
    if (inputs.isNightly) return 'experimental';
    return 'stable';
}

export function isReleaseTrack(value: string): value is ReleaseTrack {
    return (RELEASE_TRACKS as readonly string[]).includes(value);
}
