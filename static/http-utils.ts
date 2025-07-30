// Copyright (c) 2025, Compiler Explorer Authors
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

import {SentryCapture} from './sentry.js';

/**
 * HTTP utilities with intelligent Sentry error filtering.
 *
 * These utilities distinguish between:
 * - Network errors: DNS failures, connection timeouts, CORS blocks, ad blockers
 *   → Filtered out (not actionable application bugs)
 * - HTTP errors: 4xx/5xx status codes from server responses
 *   → Captured in Sentry (actionable bugs to investigate)
 */

/**
 * Check if a fetch error should be reported to Sentry.
 *
 * Network-level failures (DNS, timeout, CORS, etc.) are filtered out because:
 * - They're not caused by application bugs
 * - They create noise in error reporting
 * - They're often caused by user environment (ad blockers, network issues)
 */
export function shouldCaptureFetchError(error: any): boolean {
    // Network errors from fetch() are typically TypeError with "fetch" in message
    if (error instanceof TypeError && error.message.includes('fetch')) {
        return false;
    }

    // Filter out AbortError from cancelled requests
    if (error.name === 'AbortError') {
        return false;
    }

    return true;
}

/**
 * Fetch with automatic Sentry error filtering.
 *
 * Error handling strategy:
 * - HTTP errors (status >= 400): Captured in Sentry with context
 * - Network errors: Logged to console, not sent to Sentry
 * - Successful responses (status < 400): No error handling needed
 */
export async function fetch(url: string, options: RequestInit = {}, context?: string): Promise<Response> {
    try {
        const response = await globalThis.fetch(url, {
            credentials: 'include',
            ...options,
            headers: {
                Accept: 'application/json',
                ...options.headers,
            },
        });

        // Only capture HTTP errors (4xx/5xx), not successful responses or redirects
        if (!response.ok && response.status >= 400) {
            const error = new Error(`HTTP ${response.status}: ${response.statusText}`);
            (error as any).response = response;
            SentryCapture(error, context || 'HTTP request failed');
        }

        return response;
    } catch (error) {
        // Only capture if it's not a network-level failure
        if (shouldCaptureFetchError(error)) {
            SentryCapture(error, context || 'HTTP request error');
        } else {
            console.debug('Network request failed:', error, context);
        }
        throw error;
    }
}

/** GET request with error filtering */
export async function get(url: string, context?: string): Promise<Response> {
    return fetch(url, {method: 'GET'}, context);
}

/** POST JSON request with error filtering */
export async function postJSON(url: string, data: any, context?: string): Promise<Response> {
    return fetch(
        url,
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        },
        context,
    );
}
