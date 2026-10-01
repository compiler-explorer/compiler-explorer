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

export interface FetchJsonOptions {
    /** Number of retry attempts after the first try (so total attempts = retries + 1). */
    retries?: number;
    /** Base backoff delay in milliseconds; doubles each retry. */
    baseDelayMs?: number;
}

const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

/**
 * Fetch JSON from a CE API endpoint with retries on transient failures.
 *
 * The languages, compilers, libraries and tools lists are lazy-loaded at
 * runtime (see #8549) rather than baked into the page, so they're newly exposed
 * to transient network blips (`fetch` rejects with `TypeError: Failed to fetch`)
 * and brief server hiccups (5xx) that would otherwise break pane initialization.
 * Browsers never retry `fetch`, so we do it here with exponential backoff.
 *
 * Only network failures and 5xx responses are retried. Client errors (4xx) and
 * malformed bodies won't change on retry, so they fail fast. A non-ok response
 * that exhausts its retries throws an Error carrying the status, rather than
 * letting an error body fall through to the caller as if it were valid data.
 */
export async function fetchApiJson<T>(url: string, options: FetchJsonOptions = {}): Promise<T> {
    const {retries = 2, baseDelayMs = 250} = options;
    let lastError: unknown;
    for (let attempt = 0; attempt <= retries; attempt++) {
        if (attempt > 0) await delay(baseDelayMs * 2 ** (attempt - 1));

        let response: Response;
        try {
            response = await fetch(url, {headers: {Accept: 'application/json'}});
        } catch (e) {
            // Network-level failure ("Failed to fetch"): retry.
            lastError = e;
            continue;
        }

        if (response.ok) return (await response.json()) as T;

        const error = new Error(`Request failed: ${response.status} ${response.statusText} for ${url}`);
        // Client errors won't succeed on retry; fail fast.
        if (response.status < 500) throw error;
        lastError = error;
    }
    throw lastError;
}
