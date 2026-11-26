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

import {describe, expect, it} from 'vitest';

import {extractShortId} from '../lib/url-utils.js';

describe('extractShortId', () => {
    it('extracts the id from a full URL', () => {
        expect(extractShortId('https://godbolt.org/z/abc')).toEqual('abc');
    });

    it('extracts the id from a path (generateShareableUrl shape)', () => {
        expect(extractShortId('/z/abc')).toEqual('abc');
    });

    it('ignores query strings', () => {
        expect(extractShortId('https://godbolt.org/z/abc?utm=foo')).toEqual('abc');
        expect(extractShortId('/z/abc?utm=foo')).toEqual('abc');
    });

    it('ignores fragments', () => {
        expect(extractShortId('https://godbolt.org/z/abc#section')).toEqual('abc');
    });

    it('ignores trailing slashes', () => {
        expect(extractShortId('https://godbolt.org/z/abc/')).toEqual('abc');
        expect(extractShortId('/z/abc/')).toEqual('abc');
    });

    it('handles a non-root httpRoot', () => {
        expect(extractShortId('https://example.com/foo/z/abc')).toEqual('abc');
    });

    it('passes a bare id through unchanged', () => {
        expect(extractShortId('abc')).toEqual('abc');
    });
});
