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
import {assert} from './assert.js';
import {asSafeVer, magic_semver} from './utils.js';

const PRERELEASE_TAGS = new Set(['beta', 'alpha']);
// Deliberately omit 'snapshot' — too ambiguous: CE configs use it both for genuine
// nightly tracks (e.g. clspv main) AND descriptively on stable releases built from an
// upstream snapshot (e.g. IBM Advance Toolchain ppc64g8 = "power64 AT12.0", a stable
// release that happens to incorporate a gcc snapshot). Compilers that need 'nightly'
// classification with a (snapshot) semver should set releaseTrack=nightly explicitly.
const NIGHTLY_TAGS = new Set(['nightly', 'main', 'master']);
const RC_PATTERN = /^rc\d*$/;

// Allowlist of recognised prerelease tokens for the semver-with-suffix path. Many CE
// compilers use the `-suffix` slot for build flavour ("4.07.1-flambda"), distro revision
// ("10.3.0-2"), git hash ("3.29.20240506-g1ea8fa8"), or vendor codename ("4.1.0-cd1") —
// none of which are release-track signals. Only treat the suffix as a prerelease if it
// matches a known prerelease keyword (optionally followed by digits).
const RECOGNISED_PRERELEASE_SEGMENT_RE = /^(rc|beta|alpha|preview|dev|pre)\d*$/;

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
 *   1. Real numbered semver with a *recognised* prerelease segment
 *      (rc / beta / alpha / preview / dev / pre, optionally with digits):
 *      → 'nightly' if isNightly is also set (the maintainer's explicit signal that
 *        this is a rolling preview rather than a one-off RC — e.g. micropython-preview).
 *      → 'prerelease' otherwise (e.g. dxc 1.8.2306-preview, RC builds).
 *      Real numbered semvers with an *unrecognised* suffix (build flavour, distro rev,
 *      git hash, vendor codename — "4.07.1-flambda", "10.3.0-2", "4.1.0-cd1") fall
 *      through to rule 2 and classify as stable.
 *   2. Real numbered semver, no recognised prerelease segment → 'stable'.
 *   3. asSafeVer maps to magic_semver.trunk (semver contains "trunk"/"main"), or the
 *      bare semver tag (after stripping outer parens) is in NIGHTLY_TAGS → 'nightly'.
 *      Note: "snapshot" is NOT in NIGHTLY_TAGS — see the comment on that constant.
 *   4. semver tag is a prerelease tag ("beta", "alpha", "rc", "rc1", ...) → 'prerelease'.
 *   5. isNightly with an empty semver → 'nightly'. The convention in CE configs is that
 *      a parenthesised tag like "(contracts)" or "(modules)" names a *specific* feature
 *      fork, while no semver at all means "the canonical nightly build, nothing fancy".
 *      This catches mainstream nightlies (wasm32clang, flangtrunk, dotnettrunk*,
 *      rustccggcc-master, ...) without needing per-compiler overrides.
 *   6. isNightly with a non-empty, non-canonical tag → 'experimental' — typically the
 *      c++ language-proposal forks like "(contracts)", "(modules)", "(P2034 lambdas)".
 *   7. Anything else → 'stable' as the safe fallback.
 *
 * Cases the heuristic genuinely can't reach should set `compiler.releaseTrack=...` in
 * the .properties file — but the rules above cover the common cases without overrides.
 */
export function inferReleaseTrack(inputs: ReleaseTrackInputs): ReleaseTrack {
    const semver = inputs.semver.toLowerCase().trim();
    const tag = stripParens(semver);
    const safe = asSafeVer(semver);
    // isMagic is true for both 'trunk' (magic max) and 'non_trunk' (the asSafeVer
    // sentinel for any non-parseable semver — "(contracts)", "nightly", etc.). Both
    // cases need rules 3+ to handle; only real numbered semvers fall into rule 1/2.
    const isMagic = safe === magic_semver.trunk || safe === magic_semver.non_trunk;

    if (inputs.isSemVer && !isMagic) {
        const prerelease = semverParser.prerelease(safe);
        const firstSegment = prerelease ? String(prerelease[0]) : '';
        if (RECOGNISED_PRERELEASE_SEGMENT_RE.test(firstSegment)) {
            return inputs.isNightly ? 'nightly' : 'prerelease';
        }
        return 'stable';
    }
    if (safe === magic_semver.trunk || NIGHTLY_TAGS.has(tag)) return 'nightly';
    if (PRERELEASE_TAGS.has(tag) || RC_PATTERN.test(tag)) return 'prerelease';
    if (inputs.isNightly && semver === '') return 'nightly';
    if (inputs.isNightly) return 'experimental';
    return 'stable';
}

export function isReleaseTrack(value: string): value is ReleaseTrack {
    return (RELEASE_TRACKS as readonly string[]).includes(value);
}

/**
 * Resolve the release track for a compiler given a raw override value (typically from
 * `props('releaseTrack', '')`) and the structural inputs to fall back on.
 *
 * The raw override is whatever the properties layer returns — if a maintainer wrote
 * `releaseTrack=true` or `releaseTrack=1` it'll have been coerced by toProperty to a
 * boolean/number, so this function rejects non-string raw values up-front with a clear
 * error rather than blowing up on `.trim()`. Empty-after-trim falls back to inference.
 * Non-empty strings must be a valid ReleaseTrack value or an assertion fires (so a
 * misconfigured `releaseTrack=stabel` fails compiler discovery loudly).
 *
 * `compilerId` is only used in error messages.
 */
export function resolveReleaseTrack(
    rawOverride: unknown,
    inputs: ReleaseTrackInputs,
    compilerId: string,
): ReleaseTrack {
    assert(
        typeof rawOverride === 'string',
        `Invalid releaseTrack value for ${compilerId}: expected a string, got ${typeof rawOverride} (${String(
            rawOverride,
        )})`,
    );
    const override = rawOverride.trim();
    if (override === '') return inferReleaseTrack(inputs);
    assert(
        isReleaseTrack(override),
        `Invalid releaseTrack "${override}" for ${compilerId}; expected one of ${RELEASE_TRACKS.join('|')}`,
    );
    return override;
}

/**
 * Mutate a CompilerInfo in place so `releaseTrack` is set to a valid value: keeps any
 * existing valid value, otherwise re-infers from the structural fields. Used to defend
 * against prediscovered JSON or remote-fetched compiler lists that pre-date this field
 * or carry a hand-edited invalid value.
 */
export function backfillReleaseTrack(compiler: {
    isSemVer: boolean;
    isNightly: boolean;
    semver: string;
    releaseTrack?: ReleaseTrack | string;
}): void {
    if (compiler.releaseTrack && isReleaseTrack(compiler.releaseTrack)) return;
    compiler.releaseTrack = inferReleaseTrack({
        isSemVer: compiler.isSemVer,
        isNightly: compiler.isNightly,
        semver: compiler.semver,
    });
}
