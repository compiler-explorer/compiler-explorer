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
import {RELEASE_TRACKS} from '../../../types/compiler.interfaces.js';
import {InstructionSetsList} from '../../../types/instructionsets.js';
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
                // Bucket by (lang, instructionSet, group) so different compiler families
                // that happen to share an arch tag (e.g. an upstream gcc release and a
                // mis-tagged cross-compiler that share major 16) don't fight for the
                // same slot. Without `group`, an HPPA cross-compiler tagged amd64 would
                // hijack the x86-64 GCC slot for its semver-major.
                const groupKey = `${c.lang}\0${c.instructionSet ?? ''}\0${c.group}`;
                let bucket = stableBuckets.get(groupKey);
                if (!bucket) {
                    bucket = new Map();
                    stableBuckets.set(groupKey, bucket);
                }
                const safe = asSafeVer(c.semver);
                // Compilers tagged 'stable' but not isSemVer can only arise via an
                // explicit `releaseTrack=stable` override on a non-semver compiler
                // (the inferReleaseTrack heuristic itself never produces this combo
                // for a non-empty semver). They share a single bucket per group,
                // last-one-wins — rare enough not to be worth a more clever scheme.
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
            instructionSet: z
                .enum(InstructionSetsList)
                .optional()
                .describe(
                    'Target architecture filter. Combine with `latestPerMajor: true` for "newest X for arch Y" ' +
                        '(e.g. `{language:"c++", instructionSet:"amd64", latestPerMajor:true}`).',
                ),
            match: z
                .string()
                .optional()
                .describe(
                    'Case-insensitive AND-of-tokens filter on id and name. Punctuation splits tokens; numeric ' +
                        'tokens match whole-word ("gcc 14.1" matches "14.1.0" not "14.10"); alphanumeric tokens ' +
                        'substring-match ("g14" matches "g142"). Literal text only — for "newest version" use ' +
                        '`latestPerMajor`; for "newest X on arch Y" prefer `instructionSet` + `latestPerMajor`.',
                ),
            maxResults: z
                .number()
                .int()
                .positive()
                .optional()
                .describe(
                    `Cap full-detail entries (default ${DEFAULT_MAX_RESULTS}). Beyond the cap, degrades to lean ` +
                        '(id+name only) with a refinement hint.',
                ),
            lean: z
                .boolean()
                .optional()
                .describe('Force id+name only, regardless of count. Useful to browse the catalog before drilling in.'),
            latestPerMajor: z
                .boolean()
                .optional()
                .describe(
                    `"Newest X" filter. By release track (${RELEASE_TRACKS.join('/')}): newest stable per ` +
                        '(language, instructionSet, semver major); all nightly + prerelease; experimental ' +
                        'skipped unless `includeExperimental: true`.',
                ),
            includeExperimental: z
                .boolean()
                .optional()
                .describe(
                    'With `latestPerMajor: true`, also include experimental compilers (c++ proposal forks, ' +
                        'llvm-mos platform variants). Off by default — bloats "newest X" answers.',
                ),
        },
        async ({language, instructionSet, match, maxResults, lean, latestPerMajor, includeExperimental}) => {
            let pool = apiHandler.compilers;
            if (language) pool = pool.filter(c => c.lang === language);
            if (instructionSet) pool = pool.filter(c => c.instructionSet === instructionSet);
            const matched = applyMatch(pool, match, c => [c.id, c.name]);
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
                    // releaseTrack is one of: "stable" | "nightly" | "prerelease" | "experimental".
                    releaseTrack: c.releaseTrack,
                    // supportsExecute / supportsBinary tell the caller whether `compile`
                    // with execute=true is going to work for this compiler — saves a
                    // failed call where the agent has to try and read the error.
                    supportsExecute: c.supportsExecute ?? false,
                    supportsBinary: c.supportsBinary ?? false,
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
