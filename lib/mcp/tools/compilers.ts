// Copyright (C) 2026 Hudson River Trading LLC <opensource@hudson-trading.com>
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

import type {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import semverParser from 'semver';
import {z} from 'zod';

import type {CompilerInfo} from '../../../types/compiler.interfaces.js';
import type {ApiHandler} from '../../handlers/api.js';
import {asSafeVer} from '../../utils.js';
import {applyCap, applyMatch} from '../utils.js';

const DEFAULT_MAX_RESULTS = 25;

// Distil the "what's the newest X" view of a compiler list.
//
// `releaseTrack` (added by PR #8685) lets us treat each track on its own terms:
// - 'stable':       group by (lang, instructionSet, semver major); take the newest per major
// - 'nightly':      include every entry (one canonical nightly per language/arch is the norm,
//                   but distinct tracks like rust nightly + rust master both belong)
// - 'prerelease':   include every entry (rust beta, dxc preview etc. are first-class)
// - 'experimental': skip by default; include all when includeExperimental is set
//
// We also report droppedExperimental so the caller knows when there's a sea of feature
// forks they're not seeing — the LLM-facing hint in the response surfaces this.
function pickLatest(
    compilers: CompilerInfo[],
    includeExperimental: boolean,
): {kept: CompilerInfo[]; droppedExperimental: number} {
    let droppedExperimental = 0;
    const stableBuckets = new Map<string, Map<number, CompilerInfo>>();
    const nightlyAndPrerelease: CompilerInfo[] = [];
    const experimentals: CompilerInfo[] = [];

    for (const c of compilers) {
        switch (c.releaseTrack) {
            case 'stable': {
                const groupKey = `${c.lang}\0${c.instructionSet ?? ''}`;
                let bucket = stableBuckets.get(groupKey);
                if (!bucket) {
                    bucket = new Map();
                    stableBuckets.set(groupKey, bucket);
                }
                const safe = asSafeVer(c.semver);
                // Compilers tagged 'stable' that don't have a parseable numeric semver
                // (rare — fallback path in the inferReleaseTrack heuristic) share a
                // single bucket per group, last-one-wins. Not worth optimising further.
                const major = c.isSemVer ? semverParser.major(safe) : -1;
                const existing = bucket.get(major);
                if (!existing || semverParser.compare(safe, asSafeVer(existing.semver), true) > 0) {
                    bucket.set(major, c);
                }
                break;
            }
            case 'nightly':
            case 'prerelease':
                nightlyAndPrerelease.push(c);
                break;
            case 'experimental':
                if (includeExperimental) experimentals.push(c);
                else droppedExperimental += 1;
                break;
        }
    }

    const kept: CompilerInfo[] = [];
    for (const bucket of stableBuckets.values()) for (const c of bucket.values()) kept.push(c);
    kept.push(...nightlyAndPrerelease, ...experimentals);
    kept.sort((a, b) => {
        if (a.lang !== b.lang) return a.lang.localeCompare(b.lang);
        const aIs = a.instructionSet ?? '';
        const bIs = b.instructionSet ?? '';
        if (aIs !== bIs) return aIs.localeCompare(bIs);
        return semverParser.compare(asSafeVer(b.semver), asSafeVer(a.semver), true);
    });
    return {kept, droppedExperimental};
}

export function registerCompilersTool(server: McpServer, apiHandler: ApiHandler): void {
    server.tool(
        'list_compilers',
        'List available compilers, optionally filtered by language',
        {
            language: z.string().optional().describe('Language ID to filter by (e.g. "c++", "rust", "python")'),
            match: z
                .string()
                .optional()
                .describe(
                    'Case-insensitive AND-of-words filter on compiler id and name. The pattern is split on ' +
                        'whitespace and punctuation, and every token must appear (in any order) in the id or name. ' +
                        '"x86-64 gcc trunk" matches "x86-64 gcc (trunk)". Note: this is a literal text filter — ' +
                        '"gcc 15" excludes gcc 16.x, so to find the newest version use `latestPerMajor: true` or ' +
                        'browse with `lean: true`. For broad searches that match hundreds of compilers, the ' +
                        'response degrades to lean mode (id+name only); refine further or use `lean: true` ' +
                        'explicitly.',
                ),
            maxResults: z
                .number()
                .int()
                .positive()
                .optional()
                .describe(
                    `Maximum entries to return in the full response shape (default ${DEFAULT_MAX_RESULTS}). ` +
                        'When the filtered count exceeds this, the response degrades to lean mode (id+name only) ' +
                        'and returns ALL matches; raise this if you need full per-entry detail across many results.',
                ),
            lean: z
                .boolean()
                .optional()
                .describe(
                    'Return id+name only for every result, regardless of count. Use this to browse the catalog ' +
                        'index without risking an oversized response, then call again with an exact `match` to get ' +
                        'full details (compilerType, semver, instructionSet).',
                ),
            latestPerMajor: z
                .boolean()
                .optional()
                .describe(
                    'Distil to "what is the newest X" — the right way to find e.g. the newest GCC. Returns: the ' +
                        'newest stable per (language, instruction set, semver major); every nightly track (rust ' +
                        'nightly, gcc snapshot, ...); every prerelease track (rust beta, dxc preview, ...). ' +
                        'Experimental forks (gcc contracts/modules/coroutines branches, etc.) are skipped by ' +
                        'default; pass `includeExperimental: true` to include them.',
                ),
            includeExperimental: z
                .boolean()
                .optional()
                .describe(
                    'Only meaningful with `latestPerMajor: true`. When true, also include experimental compilers ' +
                        '(c++ language-proposal forks like gcc-contracts-trunk, llvm-mos platform variants) ' +
                        'in the result. They are skipped by default because they bloat the answer to "what is ' +
                        'the newest X" without being release-track-comparable.',
                ),
        },
        async ({language, match, maxResults, lean, latestPerMajor, includeExperimental}) => {
            const byLang = language ? apiHandler.compilers.filter(c => c.lang === language) : apiHandler.compilers;
            const matched = applyMatch(byLang, match, c => [c.id, c.name]);
            let filtered = matched;
            let droppedExperimental = 0;
            if (latestPerMajor) {
                const picked = pickLatest(matched, includeExperimental === true);
                filtered = picked.kept;
                droppedExperimental = picked.droppedExperimental;
            }
            const {items, ...meta} = applyCap(
                filtered,
                maxResults ?? DEFAULT_MAX_RESULTS,
                c => ({
                    id: c.id,
                    name: c.name,
                    lang: c.lang,
                    compilerType: c.compilerType,
                    semver: c.semver,
                    instructionSet: c.instructionSet,
                    releaseTrack: c.releaseTrack,
                }),
                'compilers',
                undefined,
                lean === true,
            );
            const response: Record<string, unknown> = {compilers: items, ...meta};
            if (latestPerMajor && droppedExperimental > 0) {
                response.droppedExperimental = droppedExperimental;
                response.latestHint =
                    `${droppedExperimental} experimental compiler(s) were skipped (feature forks, niche targets). ` +
                    'Pass `includeExperimental: true` to include them, or use `match` to find a specific one.';
            }
            return {content: [{type: 'text', text: JSON.stringify(response, null, 2)}]};
        },
    );
}
