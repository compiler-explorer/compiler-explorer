// Copyright (c) 2024, Compiler Explorer Authors
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

import zlib from 'zlib';

import {describe, expect, it} from 'vitest';

import {extractJsonFromBufferAndInflateIfRequired} from '../../lib/handlers/route-api.js';

function possibleCompression(buffer: Buffer): boolean {
    // code used in extractJsonFromBufferAndInflateIfRequired
    // required here to check criticality of test cases
    const firstByte = buffer.at(0); // for uncompressed data this is probably '{'
    return firstByte !== undefined && (firstByte & 0x0f) === 0x8; // https://datatracker.ietf.org/doc/html/rfc1950, https://datatracker.ietf.org/doc/html/rfc1950, for '{' this yields 11
}

describe('extractJsonFromBufferAndInflateIfRequired test cases', () => {
    it('check that data extraction works (good case, no compression)', () => {
        const buffer = Buffer.from('{"a":"test","b":1}');
        expect(possibleCompression(buffer)).toBeFalsy();
        const data = extractJsonFromBufferAndInflateIfRequired(buffer);
        expect(data.a).toBe('test');
        expect(data.b).toBe(1);
    });
    it('check that data extraction works (crirical case - first char indicates possible compression, no compression)', () => {
        const buffer = Buffer.from('810');
        expect(possibleCompression(buffer)).toBeTruthy();
        const data = extractJsonFromBufferAndInflateIfRequired(buffer);
        expect(data).toBe(810);
    });
    it('check that data extraction works (good case, with compression)', () => {
        const text = '{"a":"test test test test test test test test test test test test test","b":1}';
        const buffer = zlib.deflateSync(Buffer.from(text), {level: 9});
        expect(buffer.length).lessThan(text.length);
        expect(possibleCompression(buffer)).toBeTruthy();
        const data = extractJsonFromBufferAndInflateIfRequired(buffer);
        expect(data.a).toBe('test test test test test test test test test test test test test');
        expect(data.b).toBe(1);
    });
    it('check that data extraction fails (bad case)', () => {
        const buffer = Buffer.from('no json');
        expect(() => extractJsonFromBufferAndInflateIfRequired(buffer)).toThrow();
    });
});
