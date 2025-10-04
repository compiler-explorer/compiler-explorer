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
 * HTTP utility with clear error classification.
 *
 * Distinguishes between:
 * - Network errors: DNS failures, timeouts, CORS, ad blockers (not sent to Sentry)
 * - HTTP errors: 4xx/5xx status codes (sent to Sentry if context provided)
 * - Success: 2xx/3xx responses with parsed data
 */

export interface FetchResult<T = unknown> {
    /** Parsed response data (only present on success) */
    data?: T;
    /** Raw Response object (present on HTTP success/error) */
    response?: Response;
    /** Error object (present on any failure) */
    error?: Error;
    /** True if error was network-level (DNS, timeout, CORS, etc.) */
    isNetworkError: boolean;
    /** True if error was HTTP-level (4xx, 5xx status codes) */
    isHttpError: boolean;
}

interface SafeFetchOptions extends Omit<RequestInit, 'body'> {
    parseAs?: 'json' | 'text' | 'response';
    body?: BodyInit | Record<string, any> | null;
}

// Simple type-safe overloads
export function safeFetch(
    url: string,
    options: {parseAs: 'text'} & SafeFetchOptions,
    context?: string,
): Promise<FetchResult<string>>;
export function safeFetch(
    url: string,
    options: {parseAs: 'json'} & SafeFetchOptions,
    context?: string,
): Promise<FetchResult<unknown>>;
export function safeFetch(
    url: string,
    options: {parseAs: 'response'} & SafeFetchOptions,
    context?: string,
): Promise<FetchResult<Response>>;
export function safeFetch<T>(
    url: string,
    options: {parseAs: 'json'} & SafeFetchOptions,
    context?: string,
): Promise<FetchResult<T>>;
export function safeFetch<ParseAs extends string, T>(
    url: string,
    options: {parseAs: ParseAs} & SafeFetchOptions,
    context?: string,
): Promise<FetchResult<T>>;
export function safeFetch<T = string>(
    url: string,
    options?: SafeFetchOptions,
    context?: string,
): Promise<FetchResult<T>>;

export async function safeFetch(url: string, options?: SafeFetchOptions, context?: string): Promise<FetchResult<any>> {
    const {parseAs, ...fetchOptions} = options || {};

    // Auto-stringify object bodies and set Content-Type
    let processedBody = fetchOptions.body;
    const headers: HeadersInit = {...fetchOptions.headers};

    if (
        processedBody &&
        typeof processedBody === 'object' &&
        !(processedBody instanceof FormData) &&
        !(processedBody instanceof URLSearchParams) &&
        !(processedBody instanceof ReadableStream) &&
        !(processedBody instanceof ArrayBuffer) &&
        !(processedBody instanceof Blob)
    ) {
        processedBody = JSON.stringify(processedBody);
        if (!headers['Content-Type'] && !headers['content-type']) {
            headers['Content-Type'] = 'application/json';
        }
    }

    // Smart Accept header based on parseAs
    if (parseAs === 'json' && !headers['Accept'] && !headers['accept']) {
        headers['Accept'] = 'application/json';
    } else if (parseAs === 'text' && !headers['Accept'] && !headers['accept']) {
        headers['Accept'] = 'text/plain';
    }

    try {
        const response = await globalThis.fetch(url, {
            credentials: 'include',
            headers,
            ...fetchOptions,
            body: processedBody,
        });

        if (response.ok) {
            let data: any;
            try {
                if (parseAs === 'json') {
                    data = await response.json();
                } else if (parseAs === 'text') {
                    data = await response.text();
                } else if (parseAs === 'response') {
                    data = response;
                } else {
                    // Default to text for backward compatibility
                    data = await response.text();
                }
            } catch (parseError) {
                const error = new Error(
                    `Failed to parse response as ${parseAs || 'text'}: ${parseError instanceof Error ? parseError.message : String(parseError)}`,
                );
                if (context) SentryCapture(error, context);
                return {error, response, isNetworkError: false, isHttpError: false};
            }

            return {data, response, isNetworkError: false, isHttpError: false};
        } else {
            // HTTP error (4xx, 5xx)
            const error = new Error(`HTTP ${response.status}: ${response.statusText}`);
            (error as any).response = response;

            if (context) {
                SentryCapture(error, context);
            }

            return {error, response, isNetworkError: false, isHttpError: true};
        }
    } catch (fetchError) {
        // Network error (DNS, timeout, CORS, ad blocker, etc.)
        const error = fetchError as Error;
        const isNetworkError = error instanceof TypeError || error.name === 'AbortError';

        if (!isNetworkError && context) {
            // Unexpected error type - report to Sentry
            SentryCapture(error, context);
        } else if (isNetworkError) {
            // Network error - just log locally
            console.debug('Network request failed:', error.message, context);
        }

        return {error, isNetworkError, isHttpError: false};
    }
}
