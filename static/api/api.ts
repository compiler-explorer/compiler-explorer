// Copyright (c) 2021, Compiler Explorer Authors
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

import _ from 'underscore';

import {
    AssemblyDocumentationRequest,
    AssemblyDocumentationResponse,
} from '../../types/features/assembly-documentation.interfaces.js';
import {SentryCapture} from '../sentry.js';
import {FormattingRequest, FormattingResponse} from './formatting.interfaces.js';

/** Type wrapper allowing .json() to resolve to a concrete type */
interface TypedResponse<T> extends Response {
    json(): Promise<T>;
}

/** Lightweight fetch() wrapper for CE API urls */
const request = async <R>(uri: string, options?: RequestInit): Promise<TypedResponse<R>> =>
    fetch(`${window.location.origin}${window.httpRoot}api${uri}`, {
        ...options,
        credentials: 'include',
        headers: {
            ...options?.headers,
            Accept: 'application/json',
        },
    });

/** GET /api/asm/:arch/:instruction */
export const getAssemblyDocumentation = async (options: AssemblyDocumentationRequest) =>
    await request<AssemblyDocumentationResponse>(`/asm/${options.instructionSet}/${options.opcode}`);

/** POST /api/format/:formatter */
export const getFormattedCode = async (options: FormattingRequest) =>
    await request<FormattingResponse>(`/format/${options.formatterId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(_.pick(options, 'source', 'base', 'tabWidth', 'useSpaces')),
    });

/** Check if a fetch error should be reported to Sentry */
export function shouldCaptureFetchError(error: any): boolean {
    // Filter out network-level failures that aren't actionable bugs
    if (error instanceof TypeError && error.message.includes('fetch')) {
        // Network errors from fetch() are typically TypeError with "fetch" in message
        return false;
    }

    // Filter out AbortError from cancelled requests
    if (error.name === 'AbortError') {
        return false;
    }

    return true;
}

/** Fetch with automatic Sentry error filtering and retry logic */
export async function fetchWithSentryHandling(
    url: string,
    options: RequestInit = {},
    context?: string,
): Promise<Response> {
    try {
        const response = await fetch(url, {
            credentials: 'include',
            ...options,
            headers: {
                Accept: 'application/json',
                ...options.headers,
            },
        });

        // Only capture server errors (4xx/5xx) not network issues
        if (!response.ok && response.status >= 400) {
            const error = new Error(`HTTP ${response.status}: ${response.statusText}`);
            (error as any).response = response;
            SentryCapture(error, context || 'fetch request failed');
        }

        return response;
    } catch (error) {
        // Only capture if it's not a network-level failure
        if (shouldCaptureFetchError(error)) {
            SentryCapture(error, context || 'fetch request error');
        } else {
            console.debug('Network request failed:', error, context);
        }
        throw error;
    }
}

/** GET request with Sentry error handling */
export async function getWithSentryHandling(url: string, context?: string): Promise<Response> {
    return fetchWithSentryHandling(url, {method: 'GET'}, context);
}

/** POST JSON request with Sentry error handling */
export async function postJSONWithSentryHandling(url: string, data: any, context?: string): Promise<Response> {
    return fetchWithSentryHandling(
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
