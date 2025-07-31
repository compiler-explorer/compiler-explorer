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

import {beforeEach, describe, expect, it, vi} from 'vitest';
import createFetchMock from 'vitest-fetch-mock';

import {FetchResult, safeFetch} from '../http-utils.js';
import {SentryCapture} from '../sentry.js';

const fetch = createFetchMock(vi);
fetch.enableMocks();

// Mock SentryCapture
vi.mock('../sentry.js', () => ({
    SentryCapture: vi.fn(),
}));

// Mock console.debug to avoid noise in tests
vi.spyOn(console, 'debug').mockImplementation(() => {});

describe('safeFetch HTTP Utility', () => {
    beforeEach(() => {
        fetch.resetMocks();
        vi.clearAllMocks();
    });

    describe('successful requests', () => {
        it('should return data for successful text requests', async () => {
            fetch.mockResponseOnce('<h1>Hello World</h1>', {status: 200});

            const result = await safeFetch('https://example.com/page.html', {parseAs: 'text'});

            expect(result.data).toBe('<h1>Hello World</h1>');
            expect(result.error).toBeUndefined();
            expect(result.isNetworkError).toBe(false);
            expect(result.isHttpError).toBe(false);
            expect(result.response?.ok).toBe(true);
            expect(vi.mocked(SentryCapture)).not.toHaveBeenCalled();
        });

        it('should return data for successful JSON requests', async () => {
            const responseData = {users: [{id: 1, name: 'Alice'}], count: 1};
            fetch.mockResponseOnce(JSON.stringify(responseData), {status: 200});

            const result = await safeFetch('https://example.com/api/users', {parseAs: 'json'});

            expect(result.data).toEqual(responseData);
            expect(result.error).toBeUndefined();
            expect(result.isNetworkError).toBe(false);
            expect(result.isHttpError).toBe(false);
            expect(vi.mocked(SentryCapture)).not.toHaveBeenCalled();
        });

        it('should return Response object when parseAs is response', async () => {
            fetch.mockResponseOnce('test data', {status: 201});

            const result = await safeFetch('https://example.com/api', {parseAs: 'response'});

            expect(result.data).toBeInstanceOf(Response);
            expect(result.data?.status).toBe(201);
            expect(result.error).toBeUndefined();
            expect(result.isNetworkError).toBe(false);
            expect(result.isHttpError).toBe(false);
        });

        it('should default to text parsing when parseAs is not specified', async () => {
            fetch.mockResponseOnce('default text response', {status: 200});

            const result = await safeFetch('https://example.com/data');

            expect(result.data).toBe('default text response');
            expect(result.error).toBeUndefined();
        });

        it('should pass through custom headers and options', async () => {
            fetch.mockResponseOnce('OK', {status: 200});

            await safeFetch('https://example.com/api', {
                method: 'POST',
                headers: {'X-Custom': 'value'},
                body: 'test data',
                parseAs: 'text',
            });

            expect(fetch).toHaveBeenCalledWith('https://example.com/api', {
                credentials: 'include',
                method: 'POST',
                headers: {
                    'X-Custom': 'value',
                },
                body: 'test data',
            });
        });
    });

    describe('HTTP errors (4xx, 5xx)', () => {
        it('should handle 404 errors with Sentry capture', async () => {
            fetch.mockResponseOnce('Not Found', {status: 404, statusText: 'Not Found'});

            const result = await safeFetch('https://example.com/missing', {parseAs: 'text'}, 'test context');

            expect(result.data).toBeUndefined();
            expect(result.error?.message).toBe('HTTP 404: Not Found');
            expect(result.isNetworkError).toBe(false);
            expect(result.isHttpError).toBe(true);
            expect(result.response?.status).toBe(404);
            expect(vi.mocked(SentryCapture)).toHaveBeenCalledWith(
                expect.objectContaining({
                    message: 'HTTP 404: Not Found',
                    response: expect.objectContaining({status: 404}),
                }),
                'test context',
            );
        });

        it('should handle 500 errors with Sentry capture', async () => {
            fetch.mockResponseOnce('Server Error', {status: 500, statusText: 'Internal Server Error'});

            const result = await safeFetch('https://example.com/error', undefined, 'server error');

            expect(result.error?.message).toBe('HTTP 500: Internal Server Error');
            expect(result.isHttpError).toBe(true);
            expect(vi.mocked(SentryCapture)).toHaveBeenCalledWith(
                expect.objectContaining({message: 'HTTP 500: Internal Server Error'}),
                'server error',
            );
        });

        it('should not capture HTTP errors when no context provided', async () => {
            fetch.mockResponseOnce('Bad Request', {status: 400, statusText: 'Bad Request'});

            const result = await safeFetch('https://example.com/bad');

            expect(result.isHttpError).toBe(true);
            expect(result.error?.message).toBe('HTTP 400: Bad Request');
            expect(vi.mocked(SentryCapture)).not.toHaveBeenCalled();
        });
    });

    describe('network errors', () => {
        it('should handle network errors (TypeError) without Sentry capture', async () => {
            fetch.mockRejectOnce(new TypeError('Failed to fetch'));

            const result = await safeFetch('https://example.com/unreachable', {parseAs: 'json'}, 'network test');

            expect(result.data).toBeUndefined();
            expect(result.response).toBeUndefined();
            expect(result.error?.message).toBe('Failed to fetch');
            expect(result.isNetworkError).toBe(true);
            expect(result.isHttpError).toBe(false);
            expect(vi.mocked(SentryCapture)).not.toHaveBeenCalled();
            expect(console.debug).toHaveBeenCalledWith('Network request failed:', 'Failed to fetch', 'network test');
        });

        it('should handle AbortError without Sentry capture', async () => {
            const abortError = new Error('Request aborted');
            abortError.name = 'AbortError';
            fetch.mockRejectOnce(abortError);

            const result = await safeFetch('https://example.com/aborted', undefined, 'abort test');

            expect(result.error?.message).toBe('Request aborted');
            expect(result.isNetworkError).toBe(true);
            expect(result.isHttpError).toBe(false);
            expect(vi.mocked(SentryCapture)).not.toHaveBeenCalled();
            expect(console.debug).toHaveBeenCalledWith('Network request failed:', 'Request aborted', 'abort test');
        });

        it('should capture unexpected error types to Sentry', async () => {
            const unexpectedError = new Error('Unexpected error');
            unexpectedError.name = 'UnknownError';
            fetch.mockRejectOnce(unexpectedError);

            const result = await safeFetch('https://example.com/weird', undefined, 'unexpected error test');

            expect(result.error?.message).toBe('Unexpected error');
            expect(result.isNetworkError).toBe(false);
            expect(result.isHttpError).toBe(false);
            expect(vi.mocked(SentryCapture)).toHaveBeenCalledWith(unexpectedError, 'unexpected error test');
        });
    });

    describe('parsing errors', () => {
        it('should handle JSON parsing errors', async () => {
            fetch.mockResponseOnce('Invalid JSON{', {status: 200});

            const result = await safeFetch('https://example.com/bad-json', {parseAs: 'json'}, 'json parse test');

            expect(result.data).toBeUndefined();
            expect(result.error?.message).toContain('Failed to parse response as json');
            expect(result.isNetworkError).toBe(false);
            expect(result.isHttpError).toBe(false);
            expect(result.response?.ok).toBe(true);
            expect(vi.mocked(SentryCapture)).toHaveBeenCalledWith(
                expect.objectContaining({message: expect.stringContaining('Failed to parse response as json')}),
                'json parse test',
            );
        });

        it('should not capture parsing errors when no context provided', async () => {
            fetch.mockResponseOnce('Invalid JSON{', {status: 200});

            const result = await safeFetch('https://example.com/bad-json', {parseAs: 'json'});

            expect(result.error?.message).toContain('Failed to parse response as json');
            expect(vi.mocked(SentryCapture)).not.toHaveBeenCalled();
        });
    });

    describe('type safety and generics', () => {
        interface User {
            id: number;
            name: string;
        }

        it('should support TypeScript generics for JSON responses', async () => {
            const userData: User = {id: 1, name: 'Alice'};
            fetch.mockResponseOnce(JSON.stringify(userData), {status: 200});

            const result: FetchResult<User> = await safeFetch<'json', User>('https://example.com/user', {
                parseAs: 'json',
            });

            expect(result.data?.id).toBe(1);
            expect(result.data?.name).toBe('Alice');
        });

        it('should support TypeScript generics for text responses', async () => {
            fetch.mockResponseOnce('Hello World', {status: 200});

            const result: FetchResult<string> = await safeFetch<'text', string>('https://example.com/text', {
                parseAs: 'text',
            });

            expect(result.data).toBe('Hello World');
        });

        it('should correctly infer types from function overloads', async () => {
            // Test that each overload returns the correct type

            // JSON overload should return unknown
            fetch.mockResponseOnce('{"test": true}', {status: 200});
            const jsonResult = await safeFetch('https://example.com/json', {parseAs: 'json'});
            expect(typeof jsonResult.data).toBe('object');

            // Text overload should return string
            fetch.mockResponseOnce('text response', {status: 200});
            const textResult = await safeFetch('https://example.com/text', {parseAs: 'text'});
            if (textResult.data) {
                // This should only compile if textResult.data is string
                const length = textResult.data.length;
                expect(length).toBe(13);
            }

            // Response overload should return Response
            fetch.mockResponseOnce('response', {status: 200});
            const responseResult = await safeFetch('https://example.com/response', {parseAs: 'response'});
            if (responseResult.data) {
                // This should only compile if responseResult.data is Response
                expect(responseResult.data.status).toBe(200);
                expect(responseResult.data.headers).toBeInstanceOf(Headers);
            }

            // Default overload should return string
            fetch.mockResponseOnce('default', {status: 200});
            const defaultResult = await safeFetch('https://example.com/default');
            if (defaultResult.data) {
                // This should only compile if defaultResult.data is string
                const trimmed = defaultResult.data.trim();
                expect(trimmed).toBe('default');
            }
        });
    });

    describe('edge cases', () => {
        it('should handle empty response body', async () => {
            fetch.mockResponseOnce('', {status: 204});

            const result = await safeFetch('https://example.com/empty', {parseAs: 'text'});

            expect(result.data).toBe('');
            expect(result.error).toBeUndefined();
            expect(result.isNetworkError).toBe(false);
            expect(result.isHttpError).toBe(false);
        });

        it('should handle null/undefined options', async () => {
            fetch.mockResponseOnce('test', {status: 200});

            const result = await safeFetch('https://example.com/test', undefined);

            expect(result.data).toBe('test');
            expect(fetch).toHaveBeenCalledWith('https://example.com/test', {
                credentials: 'include',
                headers: {Accept: 'application/json'},
            });
        });

        it('should preserve existing headers while adding defaults', async () => {
            fetch.mockResponseOnce('OK', {status: 200});

            await safeFetch('https://example.com/api', {
                headers: {
                    Accept: 'text/plain',
                    'X-Custom': 'override',
                },
            });

            expect(fetch).toHaveBeenCalledWith('https://example.com/api', {
                credentials: 'include',
                headers: {
                    Accept: 'text/plain', // User header overrides default
                    'X-Custom': 'override',
                },
            });
        });
    });
});
