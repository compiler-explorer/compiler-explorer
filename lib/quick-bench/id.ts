// Copyright (c) 2026, Compiler Explorer Authors
// Portions derived from Quick Bench (https://github.com/FredTingaud/quick-bench-back-end,
// https://github.com/FredTingaud/quick-bench-front-end, https://github.com/FredTingaud/quick-bench-docker),
// Copyright (c) 2017, Fred Tingaud, used under the BSD 2-Clause License.
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

// A Quick Bench link id is the SHA-1 of the benchmark's code and options, base64 encoded with '/' swapped for '-',
// '+' swapped for '_', and the single '=' of padding dropped: always 27 characters. The SHA-1 is also the name of
// the stored files, so decoding an id is all it takes to find them.
//
// Quick Bench's own decoder turns '_' into "+ " and relies on Node's base64 decoder skipping the stray space; this
// one maps the characters back exactly instead.

const QUICK_BENCH_ID_RE = /^[A-Za-z0-9_-]{26}[AEIMQUYcgkosw048]$/;
const SHA1_HEX_RE = /^[0-9a-f]{40}$/;

/** Whether `id` has the shape of a Quick Bench link id. */
export function isQuickBenchId(id: string): boolean {
    return QUICK_BENCH_ID_RE.test(id);
}

/** Decodes a Quick Bench link id to the lower-case hex SHA-1 it stands for, or undefined if it isn't one. */
export function decodeQuickBenchId(id: string): string | undefined {
    if (!isQuickBenchId(id)) return undefined;
    const base64 = id.replaceAll('-', '/').replaceAll('_', '+') + '=';
    return Buffer.from(base64, 'base64').toString('hex');
}

/** Encodes a hex SHA-1 as a Quick Bench link id. */
export function encodeQuickBenchId(sha1Hex: string): string {
    const hex = sha1Hex.toLowerCase();
    if (!SHA1_HEX_RE.test(hex)) throw new Error(`Not a SHA-1: ${sha1Hex}`);
    return Buffer.from(hex, 'hex').toString('base64').replaceAll('/', '-').replaceAll('+', '_').slice(0, -1);
}
