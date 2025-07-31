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
 * Type-safe HTTP utility with clear error classification and enforced parse type relationships.
 *
 * Distinguishes between:
 * - Network errors: DNS failures, timeouts, CORS, ad blockers (not sent to Sentry)
 * - HTTP errors: 4xx/5xx status codes (sent to Sentry if context provided)
 * - Success: 2xx/3xx responses with parsed data
 */

// ===== TYPE DEFINITIONS =====

/**
 * Template literal type that creates descriptive parse option labels.
 * This helps create better error messages and IntelliSense support.
 *
 * Benefits:
 * - Clear error messages when wrong parseAs is used
 * - Better autocomplete suggestions
 * - Type-safe string operations
 */
type ParseAsOption = 'json' | 'text' | 'response';

/**
 * Conditional type that maps parseAs options to their corresponding return types.
 * This is the core of our type safety - it creates a relationship between
 * the parseAs parameter and the expected return type.
 *
 * How it works:
 * - If ParseAs extends 'json', return type is unknown (could be any JSON)
 * - If ParseAs extends 'text', return type is string
 * - If ParseAs extends 'response', return type is Response
 * - Otherwise (never case), return never
 *
 * Benefits:
 * - Eliminates unsafe type assertions
 * - Prevents mismatched parseAs/return type combinations
 * - Provides accurate IntelliSense for each parse option
 */
type ParsedDataType<ParseAs extends ParseAsOption> = ParseAs extends 'json'
    ? unknown // JSON can be any valid JSON value
    : ParseAs extends 'text'
      ? string
      : ParseAs extends 'response'
        ? Response
        : never;

/**
 * Enhanced FetchResult interface with proper generic constraints.
 * Unlike the original, T is now properly constrained to match the parseAs option.
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

/**
 * Base options interface that extends RequestInit with our parseAs option.
 * We use a generic here to capture the specific parseAs value at the type level.
 */
interface SafeFetchOptions<ParseAs extends ParseAsOption> extends RequestInit {
    parseAs: ParseAs;
}

/**
 * Options interface for backward compatibility when no parseAs is specified.
 * Defaults to text parsing behavior.
 */
interface SafeFetchOptionsDefault extends RequestInit {
    parseAs?: undefined;
}

// ===== FUNCTION OVERLOADS =====

/**
 * Function overloads enforce the correct relationship between parseAs and return type.
 * Each overload specifies exactly what combination is allowed, preventing type mismatches.
 *
 * Why overloads instead of a single generic function:
 * - Overloads provide better error messages
 * - They prevent impossible combinations at compile time
 * - Each overload can have specific documentation
 * - IntelliSense shows exactly what's expected for each variant
 */

/**
 * Overload for JSON parsing - returns unknown type that should be narrowed by caller.
 *
 * Example usage:
 *   const result = await safeFetch(url, { parseAs: 'json' });
 *   // result.data is unknown, needs type narrowing or assertion
 */
export function safeFetch(
    url: string,
    options: SafeFetchOptions<'json'>,
    context?: string,
): Promise<FetchResult<unknown>>;

/**
 * Overload for text parsing - returns string type.
 *
 * Example usage:
 *   const result = await safeFetch(url, { parseAs: 'text' });
 *   // result.data is string | undefined
 */
export function safeFetch(
    url: string,
    options: SafeFetchOptions<'text'>,
    context?: string,
): Promise<FetchResult<string>>;

/**
 * Overload for response parsing - returns Response object.
 *
 * Example usage:
 *   const result = await safeFetch(url, { parseAs: 'response' });
 *   // result.data is Response | undefined
 */
export function safeFetch(
    url: string,
    options: SafeFetchOptions<'response'>,
    context?: string,
): Promise<FetchResult<Response>>;

/**
 * Overload for backward compatibility - when no parseAs is specified, defaults to text.
 *
 * Example usage:
 *   const result = await safeFetch(url);
 *   // result.data is string | undefined (backward compatible)
 */
export function safeFetch(
    url: string,
    options?: SafeFetchOptionsDefault,
    context?: string,
): Promise<FetchResult<string>>;

/**
 * Generic overload for advanced usage where the caller wants to specify both parseAs and expected type.
 * This is constrained so that T must match what parseAs would return.
 *
 * This overload allows for cases where you know more about the JSON structure:
 *   interface User { name: string; id: number; }
 *   const result = await safeFetch<User>(url, { parseAs: 'json' });
 *   // result.data is User | undefined
 *
 * The constraint ensures T extends ParsedDataType<ParseAs>, preventing mismatches.
 */
export function safeFetch<ParseAs extends ParseAsOption, T extends ParsedDataType<ParseAs>>(
    url: string,
    options: SafeFetchOptions<ParseAs>,
    context?: string,
): Promise<FetchResult<T>>;

// ===== IMPLEMENTATION =====

/**
 * Implementation function that handles all the overloaded cases.
 * Note: This implementation signature is more permissive than the overloads,
 * which is intentional - the overloads provide the type safety at the call site.
 */
export async function safeFetch(
    url: string,
    options?: (SafeFetchOptions<ParseAsOption> | SafeFetchOptionsDefault) & RequestInit,
    context?: string,
): Promise<FetchResult<unknown>> {
    // Extract parseAs with proper default handling
    const {parseAs = undefined, ...fetchOptions} = options || {};

    try {
        const response = await globalThis.fetch(url, {
            credentials: 'include',
            headers: {
                Accept: 'application/json',
                ...fetchOptions.headers,
            },
            ...fetchOptions,
        });

        if (response.ok) {
            // Success case - parse data based on parseAs option
            // Note: We can safely use type assertions here because our overloads
            // guarantee the relationship between parseAs and expected return type
            let data: unknown;

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
                // Parsing failed - treat as error
                const error = createParseError(parseAs, parseError);
                if (context) SentryCapture(error, context);
                return {error, response, isNetworkError: false, isHttpError: false};
            }

            return {data, response, isNetworkError: false, isHttpError: false};
        } else {
            // HTTP error (4xx, 5xx)
            const error = createHttpError(response);

            if (context) {
                SentryCapture(error, context);
            }

            return {error, response, isNetworkError: false, isHttpError: true};
        }
    } catch (fetchError) {
        return handleNetworkError(fetchError, context);
    }
}

// ===== HELPER FUNCTIONS =====

/**
 * Creates a descriptive parse error with proper typing.
 * This helper improves code organization and provides consistent error messages.
 */
function createParseError(parseAs: ParseAsOption | undefined, parseError: unknown): Error {
    const parseType = parseAs || 'text';
    const message = parseError instanceof Error ? parseError.message : String(parseError);
    return new Error(`Failed to parse response as ${parseType}: ${message}`);
}

/**
 * Creates an HTTP error with attached response data.
 * Uses object property assignment instead of type assertion for clarity.
 */
function createHttpError(response: Response): Error & {response: Response} {
    const error = new Error(`HTTP ${response.status}: ${response.statusText}`) as Error & {response: Response};
    error.response = response;
    return error;
}

/**
 * Handles network errors with proper classification and logging.
 * Encapsulates the network error detection logic for reusability.
 */
function handleNetworkError(fetchError: unknown, context?: string): FetchResult<unknown> {
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
