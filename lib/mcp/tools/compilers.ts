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

// Pick the newest compiler per (lang, instructionSet, semver major). Drops compilers
// that aren't isSemVer (their version strings can't be ordered meaningfully — caller
// gets a hint listing how many were dropped). Within each (lang, instructionSet) group
// we use CE's own semver helpers (asSafeVer + semverParser.compare), the same as how
// CE itself sorts compiler dropdowns; trunk/main map to a sentinel max version.
function pickLatestPerMajor(compilers: CompilerInfo[]): {kept: CompilerInfo[]; droppedNonSemver: number} {
    const semverCompilers = compilers.filter(c => c.isSemVer);
    const droppedNonSemver = compilers.length - semverCompilers.length;
    const groups = new Map<string, Map<number, CompilerInfo>>();
    for (const c of semverCompilers) {
        const safe = asSafeVer(c.semver);
        const major = semverParser.major(safe);
        const groupKey = `${c.lang}\0${c.instructionSet ?? ''}`;
        let bucket = groups.get(groupKey);
        if (!bucket) {
            bucket = new Map();
            groups.set(groupKey, bucket);
        }
        const existing = bucket.get(major);
        if (!existing || semverParser.compare(safe, asSafeVer(existing.semver), true) > 0) {
            bucket.set(major, c);
        }
    }
    const kept: CompilerInfo[] = [];
    for (const bucket of groups.values()) {
        for (const c of bucket.values()) kept.push(c);
    }
    kept.sort((a, b) => {
        if (a.lang !== b.lang) return a.lang.localeCompare(b.lang);
        const aIs = a.instructionSet ?? '';
        const bIs = b.instructionSet ?? '';
        if (aIs !== bIs) return aIs.localeCompare(bIs);
        return semverParser.compare(asSafeVer(b.semver), asSafeVer(a.semver), true);
    });
    return {kept, droppedNonSemver};
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
                    'Return only the newest compiler for each (language, instruction set, semver major) — the ' +
                        'right way to ask "what is the newest GCC / Clang / etc". Compilers without semver ' +
                        'metadata (e.g. some MSVC builds) are dropped from this view; the response includes a ' +
                        'count of how many were excluded.',
                ),
        },
        async ({language, match, maxResults, lean, latestPerMajor}) => {
            const byLang = language ? apiHandler.compilers.filter(c => c.lang === language) : apiHandler.compilers;
            const matched = applyMatch(byLang, match, c => [c.id, c.name]);
            let filtered = matched;
            let droppedNonSemver = 0;
            if (latestPerMajor) {
                const picked = pickLatestPerMajor(matched);
                filtered = picked.kept;
                droppedNonSemver = picked.droppedNonSemver;
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
                }),
                'compilers',
                undefined,
                lean === true,
            );
            const response: Record<string, unknown> = {compilers: items, ...meta};
            if (latestPerMajor && droppedNonSemver > 0) {
                response.droppedNonSemver = droppedNonSemver;
                response.latestHint =
                    `${droppedNonSemver} compiler(s) were excluded from latestPerMajor because they lack semver ` +
                    'metadata. Drop `latestPerMajor` and use `match` to find them.';
            }
            return {content: [{type: 'text', text: JSON.stringify(response, null, 2)}]};
        },
    );
}
