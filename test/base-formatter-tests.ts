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

import {describe, expect, it} from 'vitest';

import {BaseFormatter} from '../lib/formatters/base.js';

class Formatter extends BaseFormatter {}

describe('Basic formatter functionality', () => {
    it('should be one-true-style if the styles are empty', () => {
        const fmt = new Formatter({
            name: 'foo-format',
            exe: '/dev/null',
            styles: [],
            type: 'foofmt',
            version: 'foobar-format 1.0.0',
        });
        expect(fmt.isValidStyle('foostyle')).toBe(false);
        expect(fmt.formatterInfo.styles).toEqual([]);
    });

    it('should return an array of args for formatters with styles', () => {
        const fmt = new Formatter({
            name: 'foo-format',
            exe: '/dev/null',
            styles: ['foostyle'],
            type: 'foofmt',
            version: 'foobar-format 1.0.0',
        });
        expect(fmt.isValidStyle('foostyle')).toBe(true);
        expect(fmt.formatterInfo.styles).toEqual(['foostyle']);
    });
});
