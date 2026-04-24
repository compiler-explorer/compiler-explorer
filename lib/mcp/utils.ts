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

export function applyMatch<T>(items: T[], pattern: string | undefined, extract: (item: T) => string[]): T[] {
    if (!pattern) return items;
    const needle = pattern.toLowerCase();
    return items.filter(item => extract(item).some(field => field.toLowerCase().includes(needle)));
}

export type CappedList<T> = {
    items: T[];
    total: number;
    truncated: boolean;
};

export function applyCap<T>(items: T[], maxResults: number): CappedList<T> {
    if (items.length <= maxResults) {
        return {items, total: items.length, truncated: false};
    }
    return {items: items.slice(0, maxResults), total: items.length, truncated: true};
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
