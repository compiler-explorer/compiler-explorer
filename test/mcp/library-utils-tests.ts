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

import {normaliseLibraryVersion, normaliseRequestLibraries} from '../../lib/mcp/library-utils.js';

const libraries = [
    {
        id: 'boost',
        versions: [
            {id: '188', version: '1.88.0'},
            {id: '189', version: '1.89.0'},
            {id: 'trunk', version: 'trunk'},
        ],
    },
    {id: 'fmt', versions: [{id: '1100', version: '11.0.0'}]},
];

describe('normaliseLibraryVersion', () => {
    it('returns the id unchanged when given an id', () => {
        expect(normaliseLibraryVersion(libraries, 'boost', '188')).toEqual({ok: true, version: '188'});
        expect(normaliseLibraryVersion(libraries, 'boost', 'trunk')).toEqual({ok: true, version: 'trunk'});
    });

    it('translates a human version to its id', () => {
        expect(normaliseLibraryVersion(libraries, 'boost', '1.88.0')).toEqual({ok: true, version: '188'});
        expect(normaliseLibraryVersion(libraries, 'fmt', '11.0.0')).toEqual({ok: true, version: '1100'});
    });

    it('reports unknown-library for an unknown library id', () => {
        const result = normaliseLibraryVersion(libraries, 'banana', '1.0.0');
        expect(result).toEqual({ok: false, reason: 'unknown-library'});
    });

    it('reports unknown-version with the available list', () => {
        const result = normaliseLibraryVersion(libraries, 'boost', '99.0.0');
        expect(result.ok).toBe(false);
        if (!result.ok && result.reason === 'unknown-version') {
            expect(result.available).toEqual([
                {id: '188', version: '1.88.0'},
                {id: '189', version: '1.89.0'},
                {id: 'trunk', version: 'trunk'},
            ]);
        } else {
            throw new Error('expected unknown-version failure');
        }
    });
});

describe('normaliseRequestLibraries', () => {
    it('normalises a mix of id-form and human-form requests in one call', () => {
        const result = normaliseRequestLibraries(libraries, 'c++', [
            {id: 'boost', version: '188'},
            {id: 'fmt', version: '11.0.0'},
        ]);
        expect(result).toEqual({
            ok: true,
            libraries: [
                {id: 'boost', version: '188'},
                {id: 'fmt', version: '1100'},
            ],
        });
    });

    it('passes requests through untouched when knownLibraries is empty', () => {
        const requests = [{id: 'boost', version: 'whatever'}];
        const result = normaliseRequestLibraries([], 'c++', requests);
        expect(result).toEqual({ok: true, libraries: requests});
    });

    it('returns a clean error for an unknown library, naming list_libraries', () => {
        const result = normaliseRequestLibraries(libraries, 'c++', [{id: 'banana', version: '1.0'}]);
        expect(result.ok).toBe(false);
        if (!result.ok) {
            expect(result.errorText).toMatch(/Library "banana" not found/);
            expect(result.errorText).toMatch(/list_libraries/);
        }
    });

    it('returns a clean error for an unknown version, naming sample versions', () => {
        const result = normaliseRequestLibraries(libraries, 'c++', [{id: 'boost', version: '99.0.0'}]);
        expect(result.ok).toBe(false);
        if (!result.ok) {
            expect(result.errorText).toMatch(/Version "99.0.0" not found/);
            expect(result.errorText).toMatch(/188 \(1\.88\.0\)/);
            expect(result.errorText).toMatch(/match: "boost"/);
        }
    });
});
