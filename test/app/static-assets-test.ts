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

import {describe, expect, it} from 'vitest';
import {getFaviconFilename} from '../../lib/app/server.js';
import {createDefaultPugRequireHandler} from '../../lib/app/static-assets.js';

describe('Static assets', () => {
    describe('createDefaultPugRequireHandler', () => {
        it('should handle paths with manifest', () => {
            const manifest = {
                'file1.js': 'file1.hash123.js',
            };
            const handler = createDefaultPugRequireHandler('/static', manifest);

            expect(handler('file1.js')).toBe('/static/file1.hash123.js');
            expect(handler('file2.js')).toBe(''); // Not in manifest
        });

        it('should handle paths without manifest', () => {
            const handler = createDefaultPugRequireHandler('/static');

            expect(handler('file1.js')).toBe('/static/file1.js');
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
});
