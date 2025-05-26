// Copyright (c) 2017, Compiler Explorer Authors
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

import * as google from '../lib/shortener/google.js';

const googlEndpoint = 'https://goo.gl/short';

const fetch = createFetchMock(vi);
fetch.enableMocks();

describe('Google short URL resolver tests', () => {
    beforeEach(() => {
        fetch.resetMocks();
    });

    const resolver = new google.ShortLinkResolver();

    it('Resolves simple URLs', async () => {
        fetch.mockResponse('', {status: 302, headers: {Location: 'http://long.url/'}, counter: 1});

        await expect(resolver.resolve(googlEndpoint)).resolves.toEqual({longUrl: 'http://long.url/'});
        expect(fetch.requests().length).toEqual(1);
        expect(fetch.requests()[0].url).toEqual(googlEndpoint + '?si=1');
        expect(fetch.requests()[0].method).toEqual('HEAD');
    });

    it('Handles missing long urls', async () => {
        fetch.mockResponse('', {status: 404});

        await expect(resolver.resolve(googlEndpoint)).rejects.toThrow('Got response 404');
    });

    it('Handles missing location header', async () => {
        fetch.mockResponse('', {status: 302});

        await expect(resolver.resolve(googlEndpoint)).rejects.toThrow('Missing location url');
    });

    it('Handles failed requests', async () => {
        fetch.mockReject(new Error('Something went wrong'));

        await expect(resolver.resolve(googlEndpoint)).rejects.toThrow('Something went wrong');
    });

    // Do not run this test by default, as it makes a real request.
    it.skip('Handles actual real requests', async () => {
        fetch.disableMocks();

        // This taken from https://github.com/compiler-explorer/compiler-explorer/issues/113
        await expect(resolver.resolve('https://goo.gl/F02h38')).resolves.toEqual({
            longUrl: expect.stringContaining('http://gcc.godbolt.org/#'),
        });
    });
});

describe('ShortLinkResolver utility methods', () => {
    const resolver = new google.ShortLinkResolver();

    describe('extractFragment', () => {
        it('should extract fragment from goo.gl URL', () => {
            expect(resolver.extractFragment('https://goo.gl/abcd1234')).toBe('abcd1234');
        });

        it('should extract fragment from URL with query parameters', () => {
            expect(resolver.extractFragment('https://goo.gl/xyz789?param=value')).toBe('xyz789');
        });

        it('should handle URL with multiple query parameters', () => {
            expect(resolver.extractFragment('https://goo.gl/test123?foo=bar&baz=qux')).toBe('test123');
        });

        it('should handle URL with fragment and query', () => {
            expect(resolver.extractFragment('https://goo.gl/hello?world=123')).toBe('hello');
        });

        it('should handle simple fragment', () => {
            expect(resolver.extractFragment('https://goo.gl/x')).toBe('x');
        });

        it('should handle URL ending with slash', () => {
            expect(resolver.extractFragment('https://goo.gl/fragment/')).toBe('');
        });

        it('should return empty string for empty URL', () => {
            expect(resolver.extractFragment('')).toBe('');
        });

        it('should return empty string for URL with no path segments', () => {
            expect(resolver.extractFragment('https://')).toBe('');
        });

        it('should handle URL without fragment', () => {
            expect(resolver.extractFragment('https://goo.gl/')).toBe('');
        });
    });

    describe('hasDynamoDbConfigured', () => {
        it('should return false for resolver without DynamoDB config', () => {
            const basicResolver = new google.ShortLinkResolver();
            expect(basicResolver.hasDynamoDbConfigured()).toBe(false);
        });

        it('should return false for resolver with undefined props', () => {
            const basicResolver = new google.ShortLinkResolver(undefined);
            expect(basicResolver.hasDynamoDbConfigured()).toBe(false);
        });

        it('should return false for resolver with empty props', () => {
            const mockProps = vi.fn().mockReturnValue(undefined);
            const basicResolver = new google.ShortLinkResolver(mockProps);
            expect(basicResolver.hasDynamoDbConfigured()).toBe(false);
        });
    });
});
