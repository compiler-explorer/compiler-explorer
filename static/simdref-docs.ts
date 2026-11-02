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

// Docs come from simdref.diamondinoia.com, the static site of github.com/simd-labs/simdref-web (CORS '*').
export const SIMDREF_BASE_URL = 'https://simdref.diamondinoia.com';

export type SimdrefArch = 'x86' | 'arm' | 'riscv';

export interface SimdrefDoc {
    key: string;
    summary: string;
    markdown: string;
    url: string;
}

interface SimdrefMeasurement {
    latency?: string;
    uarch?: string;
}

interface SimdrefEntry {
    summary?: string;
    mnemonic?: string;
    display_mnemonic?: string;
    isa?: string[];
    display_isa?: string;
    description?: Record<string, string>;
    measurements?: SimdrefMeasurement[];
}

type SimdrefChunk = Record<string, SimdrefEntry>;

const chunkCache = new Map<string, Promise<SimdrefChunk | null>>();

const x86RepeatPrefixes = ['lock', 'rep', 'repe', 'repne', 'repz', 'repnz'];

// The rule simdref-web's app.js uses to pick the detail chunk for a mnemonic.
function chunkPrefix(mnemonic: string): string {
    const c = mnemonic.trim().toUpperCase();
    return c.length >= 3 ? c.slice(0, 3) : c;
}

function fetchChunk(url: string): Promise<SimdrefChunk | null> {
    const cached = chunkCache.get(url);
    if (cached) return cached;
    const promise = fetch(url)
        .then(response => {
            if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
            return response.json() as Promise<SimdrefChunk>;
        })
        .catch(() => {
            // Failed chunks are not cached, so a later call retries the fetch.
            chunkCache.delete(url);
            return null;
        });
    chunkCache.set(url, promise);
    return promise;
}

function findEntry(
    chunk: SimdrefChunk,
    arch: SimdrefArch,
    mnemonic: string,
): {key: string; entry: SimdrefEntry} | null {
    const base = `${arch}:${mnemonic}`;
    const keys = Object.keys(chunk)
        .filter(k => k === base || k.startsWith(base + ' '))
        .sort();
    if (keys.length === 0) return null;
    // Skip x86 masked forms and arm placeholder forms when the mnemonic also has plain forms.
    const plain = keys.filter(k => !k.includes(' k') && !k.includes('{k') && !k.includes('<'));
    const candidates = plain.length > 0 ? plain : keys;
    const counts = new Map<string, number>();
    for (const key of candidates) {
        const summary = (chunk[key].summary ?? '').replace(/\s+/g, ' ').trim();
        counts.set(summary, (counts.get(summary) ?? 0) + 1);
    }
    let bestCount = 0;
    let bestKey = candidates[0];
    for (const key of candidates) {
        const count = counts.get((chunk[key].summary ?? '').replace(/\s+/g, ' ').trim()) ?? 0;
        if (count > bestCount) {
            bestCount = count;
            bestKey = key;
        }
    }
    return {key: bestKey, entry: chunk[bestKey]};
}

function lowestLatency(measurements: SimdrefMeasurement[] | undefined): {latency: string; uarch: string} | null {
    let best: {latency: string; uarch: string} | null = null;
    let bestValue = Number.POSITIVE_INFINITY;
    for (const measurement of measurements ?? []) {
        if (measurement.latency === undefined) continue;
        const value = parseFloat(measurement.latency);
        if (!(value < bestValue)) continue;
        bestValue = value;
        best = {latency: measurement.latency, uarch: measurement.uarch ?? ''};
    }
    return best;
}

export function simdrefArch(instructionSet: string | undefined | null): SimdrefArch | null {
    switch (instructionSet) {
        case 'amd64':
        case 'x86':
            return 'x86';
        case 'aarch64':
        case 'arm32':
            return 'arm';
        case 'riscv32':
        case 'riscv64':
            return 'riscv';
        default:
            return null;
    }
}

export function asmMnemonic(line: string): string | null {
    const trimmed = line.trim();
    if (!trimmed || trimmed.endsWith(':')) return null;
    if (
        trimmed.startsWith('.') ||
        trimmed.startsWith('#') ||
        trimmed.startsWith(';') ||
        trimmed.startsWith('//') ||
        trimmed.startsWith('@')
    ) {
        return null;
    }
    const tokens = trimmed.split(/\s+/);
    const mnemonic = x86RepeatPrefixes.indexOf(tokens[0].toLowerCase()) === -1 ? tokens[0] : tokens[1];
    return mnemonic === undefined ? null : mnemonic.toLowerCase();
}

export async function getSimdrefDoc(arch: SimdrefArch, mnemonic: string): Promise<SimdrefDoc | null> {
    const prefix = chunkPrefix(mnemonic);
    if (!prefix) return null;
    const chunk = await fetchChunk(`${SIMDREF_BASE_URL}/detail-chunks/${prefix}.json`);
    if (!chunk) return null;
    let found = findEntry(chunk, arch, mnemonic);
    if (!found && arch === 'x86' && /[bwlq]$/.test(mnemonic)) {
        // AT&T mnemonics carry a one-letter operand-size suffix that simdref's matcher drops.
        const stripped = mnemonic.slice(0, -1);
        const strippedPrefix = chunkPrefix(stripped);
        const strippedChunk =
            strippedPrefix === prefix
                ? chunk
                : await fetchChunk(`${SIMDREF_BASE_URL}/detail-chunks/${strippedPrefix}.json`);
        if (strippedChunk) found = findEntry(strippedChunk, arch, stripped);
    }
    if (!found) return null;
    const {key, entry} = found;
    const summary = (entry.summary ?? '').replace(/\s+/g, ' ').trim();
    const name = entry.display_mnemonic ?? entry.mnemonic ?? mnemonic.toUpperCase();
    const isa = entry.display_isa ?? (entry.isa ?? []).join(', ');
    const url = `${SIMDREF_BASE_URL}/#${encodeURIComponent(key)}`;
    const parts = [`**${name}** (${isa})`, '', summary];
    const description = entry.description?.Description ?? '';
    if (description) parts.push('', description.trim());
    const operation = entry.description?.Operation ?? '';
    if (operation) parts.push('', '```', operation.trim(), '```');
    const lowest = lowestLatency(entry.measurements);
    if (lowest) parts.push('', `Latency ${lowest.latency} cycles (${lowest.uarch})`);
    parts.push('', `[Full docs on simdref](${url})`);
    return {key, summary, markdown: parts.join('\n'), url};
}
