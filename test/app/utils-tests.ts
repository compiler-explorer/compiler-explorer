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

import {afterEach, beforeEach, describe, expect, it} from 'vitest';
import {getFaviconFilename, isDevMode, measureEventLoopLag, parseNumberForOptions} from '../../lib/app/utils.js';

describe('App Utils', () => {
    describe('parseNumberForOptions', () => {
        it('should parse valid numbers', () => {
            expect(parseNumberForOptions('123')).toEqual(123);
            expect(parseNumberForOptions('0')).toEqual(0);
            expect(parseNumberForOptions('-10')).toEqual(-10);
        });

        it('should throw on invalid number - abc', () => {
            expect(() => parseNumberForOptions('abc')).toThrow();
        });

        it('should throw on invalid number - 123abc', () => {
            expect(() => parseNumberForOptions('123abc')).toThrow();
        });

        it('should throw on empty string', () => {
            expect(() => parseNumberForOptions('')).toThrow();
        });
    });

    describe('isDevMode', () => {
        let originalNodeEnv: string | undefined;

        beforeEach(() => {
            originalNodeEnv = process.env.NODE_ENV;
        });

        afterEach(() => {
            process.env.NODE_ENV = originalNodeEnv;
        });

        it('should return true when NODE_ENV is not production', () => {
            process.env.NODE_ENV = 'development';
            expect(isDevMode()).toBe(true);

            process.env.NODE_ENV = '';
            expect(isDevMode()).toBe(true);

            delete process.env.NODE_ENV;
            expect(isDevMode()).toBe(true);
        });

        it('should return false when NODE_ENV is production', () => {
            process.env.NODE_ENV = 'production';
            expect(isDevMode()).toBe(false);
        });
    });

    describe('getFaviconFilename', () => {
        it('should return dev favicon when in dev mode', () => {
            expect(getFaviconFilename(true, [])).toBe('favicon-dev.ico');
            expect(getFaviconFilename(true, ['beta'])).toBe('favicon-dev.ico');
        });

        it('should return beta favicon when in beta environment', () => {
            expect(getFaviconFilename(false, ['beta'])).toBe('favicon-beta.ico');
            expect(getFaviconFilename(false, ['beta', 'other'])).toBe('favicon-beta.ico');
        });

        it('should return staging favicon when in staging environment', () => {
            expect(getFaviconFilename(false, ['staging'])).toBe('favicon-staging.ico');
            expect(getFaviconFilename(false, ['other', 'staging'])).toBe('favicon-staging.ico');
        });

        it('should return default favicon otherwise', () => {
            expect(getFaviconFilename(false, [])).toBe('favicon.ico');
            expect(getFaviconFilename(false, ['other'])).toBe('favicon.ico');
            expect(getFaviconFilename(false)).toBe('favicon.ico');
        });
    });

    describe('measureEventLoopLag', () => {
        it('should return a Promise resolving to a number', () => {
            // Just verify the function returns a Promise that resolves to a number
            // We don't test actual timing as that's environment-dependent
            return expect(measureEventLoopLag(1)).resolves.toBeTypeOf('number');
        });
    });
});
