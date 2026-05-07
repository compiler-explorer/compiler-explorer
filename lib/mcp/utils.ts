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

// Replace anything that isn't an alphanumeric or '+' (kept for "c++") with whitespace,
// then collapse runs of whitespace. Lets "x86-64 gcc trunk" match "x86-64 gcc (trunk)".
function normalise(s: string): string {
    return s
        .toLowerCase()
        .replace(/[^a-z0-9+]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

export function applyMatch<T>(items: T[], pattern: string | undefined, extract: (item: T) => string[]): T[] {
    if (!pattern) return items;
    const tokens = normalise(pattern).split(' ').filter(Boolean);
    if (tokens.length === 0) return items;
    return items.filter(item => {
        const haystack = normalise(extract(item).join(' '));
        return tokens.every(token => haystack.includes(token));
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
 * Caller-requested lean mode (`forceLean: true`) returns all items mapped through
 * `leanMap` (defaults to `{id, name}`) — useful for browsing the catalog index
 * before drilling down by exact id, without first overflowing the response.
 *
 * Otherwise, below `maxResults` returns items mapped through `fullMap` (the
 * full-detail shape). At or above the cap, degrades to lean mode automatically
 * with a `leanMode: true` marker plus an LLM-facing hint suggesting refinement.
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
    if (forceLean) {
        return {
            items: items.map(lean),
            total: items.length,
            leanMode: true,
        };
    }
    if (items.length <= maxResults) {
        return {items: items.map(fullMap), total: items.length};
    }
    return {
        items: items.map(lean),
        total: items.length,
        leanMode: true,
        hint:
            `${items.length} ${entityName} exceeded the full-detail cap of ${maxResults}; showing id and name only. ` +
            'Refine your filter (e.g. add a version or architecture), use `lean: true` explicitly to confirm ' +
            'this shape, or query again with the exact id for full details.',
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
    return {
        text: kept.map(line => line.text).join('\n'),
        truncated,
        totalLines: all.length,
    };
}
