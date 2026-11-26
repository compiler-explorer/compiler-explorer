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

import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {filterEscapeSequences} from '../utils.js';

// Absolute upper bound on items returned in lean mode. Without this, calling
// list_compilers without filters can return 1000+ entries (~88KB just for c++,
// 23k+ lines globally) and overwhelm the LLM caller's response limit. With the
// cap an unfiltered call still returns SOMETHING usable, plus a hint pushing
// the agent to refine via match / language / instructionSet / latestPerMajor.
const LEAN_HARD_CAP = 200;

// Replace anything that isn't alphanumeric, '+' (kept for "c++"), or '.' (kept so
// dotted version tokens like "14.1" stay together) with whitespace, then collapse
// runs of whitespace. Lets "x86-64 gcc trunk" match "x86-64 gcc (trunk)".
function normalise(s: string): string {
    return s
        .toLowerCase()
        .replace(/[^a-z0-9+.]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

// Numeric (or dotted-numeric) tokens get a stricter "version prefix" match so
// "14.1" finds "14.1" and "14.1.x" but NOT "14.10" or "14.0.1". Alphanumeric
// tokens still match as substrings so partial-id queries like "g14" find "g142".
const NUMERIC_TOKEN = /^\d+(?:\.\d+)*$/;

export function applyMatch<T>(items: T[], pattern: string | undefined, extract: (item: T) => string[]): T[] {
    if (!pattern) return items;
    const tokens = normalise(pattern).split(' ').filter(Boolean);
    if (tokens.length === 0) return items;
    return items.filter(item => {
        const normalised = normalise(extract(item).join(' '));
        const sentinelled = ` ${normalised} `;
        return tokens.every(token => {
            if (!NUMERIC_TOKEN.test(token)) return normalised.includes(token);
            // Numeric/dotted-numeric: token must be followed by either a separator
            // (whitespace) or another version segment (`.` then more digits). This
            // makes "14.1" match "14.1" and "14.1.0" but not "14.10" or "14.0.1".
            return sentinelled.includes(` ${token} `) || sentinelled.includes(` ${token}.`);
        });
    });
}

export type LeanShape = {id: string; name: string};

export type CappedResult<F, L = LeanShape> = {
    items: F[] | L[];
    total: number;
    leanMode?: true;
    hint?: string;
};

const defaultLeanMap = <T extends {id: string; name?: string}>(item: T): LeanShape => ({
    id: item.id,
    name: item.name ?? '',
});

/**
 * Caller-requested lean mode (`forceLean: true`) returns items mapped through
 * `leanMap` (defaults to `{id, name}`) — useful for browsing the catalog index
 * before drilling down by exact id, without first overflowing the response.
 *
 * Otherwise, below `maxResults` returns items mapped through `fullMap` (the
 * full-detail shape). At or above the cap, degrades to lean mode automatically
 * with a `leanMode: true` marker plus an LLM-facing hint suggesting refinement.
 *
 * In every lean path the response is hard-capped at `LEAN_HARD_CAP` items so a
 * very broad query still returns a useful (truncated) list rather than a
 * megabyte of JSON the host MCP client will reject. The hint always carries the
 * `total` so the caller knows refinement is needed.
 */
export function applyCap<T extends {id: string; name?: string}, F, L = LeanShape>(
    items: T[],
    maxResults: number,
    fullMap: (item: T) => F,
    entityName: string,
    leanMap?: (item: T) => L,
    forceLean = false,
): CappedResult<F, L> {
    const lean = leanMap ?? (defaultLeanMap as unknown as (item: T) => L);
    const total = items.length;
    const exceedsFullCap = total > maxResults;
    const useLean = forceLean || exceedsFullCap;

    if (!useLean) {
        return {items: items.map(fullMap), total};
    }

    const truncated = total > LEAN_HARD_CAP;
    const kept = truncated ? items.slice(0, LEAN_HARD_CAP) : items;

    let hint: string | undefined;
    if (exceedsFullCap && truncated) {
        hint =
            `${total} ${entityName} matched and the response is too large to return in full; ` +
            `showing the first ${LEAN_HARD_CAP} (id and name only). Refine with \`match\`, \`language\`, ` +
            '`instructionSet`, or `latestPerMajor` to narrow the result.';
    } else if (exceedsFullCap) {
        hint =
            `${total} ${entityName} exceeded the full-detail cap of ${maxResults}; showing id and name only. ` +
            'Refine your filter (e.g. add a version or architecture), use `lean: true` explicitly to confirm ' +
            'this shape, or query again with the exact id for full details.';
    } else if (truncated) {
        hint =
            `Lean response capped at ${LEAN_HARD_CAP} of ${total} ${entityName}; refine your filter ` +
            '(`match`, `language`, `instructionSet`) to see the rest.';
    }

    return {
        items: kept.map(lean),
        total,
        leanMode: true,
        ...(hint && {hint}),
    };
}

export type TruncatedLines = {
    text: string;
    truncated: boolean;
    totalLines: number;
};

export function truncateLines(lines: ResultLine[] | null | undefined, maxLines: number): TruncatedLines {
    const all = lines || [];
    const truncated = all.length > maxLines;
    const kept = truncated ? all.slice(0, maxLines) : all;
    // Strip ANSI escape sequences (gcc/clang colour their diagnostics; MCP
    // consumers don't have terminals so the codes are pure noise).
    return {
        text: kept.map(line => filterEscapeSequences(line.text)).join('\n'),
        truncated,
        totalLines: all.length,
    };
}
