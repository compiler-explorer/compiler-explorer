// Copyright (c) 2023, Compiler Explorer Authors
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

import * as rison from '../shared/rison.js';

// Copied from https://github.com/Nanonid/rison/blob/master/python/rison/tests.py
const py_testcases = {
    '(a:0,b:1)': {a: 0, b: 1},
    "(a:0,b:foo,c:'23skidoo')": {a: 0, c: '23skidoo', b: 'foo'},
    '!t': true,
    '!f': false,
    '!n': null,
    "''": '',
    0: 0,
    1.5: 1.5,
    '-3': -3,
    '1e30': 1e30,
    '1e-30': 1.0000000000000001e-30,
    'G.': 'G.',
    a: 'a',
    "'0a'": '0a',
    "'abc def'": 'abc def',
    '()': {},
    '(a:0)': {a: 0},
    '(id:!n,type:/common/document)': {type: '/common/document', id: null},
    '!()': [],
    "!(!t,!f,!n,'')": [true, false, null, ''],
    "'-h'": '-h',
    'a-z': 'a-z',
    "'wow!!'": 'wow!',
    'domain.com': 'domain.com',
    "'user@domain.com'": 'user@domain.com',
    "'US $10'": 'US $10',
    "'can!'t'": "can't",
};

const encode_testcases = {
    "can't": "'can!'t'",
    '"can\'t"': "'\"can!'t\"'",
    "'can't'": "'!'can!'t!''",
};

describe('Rison test cases', () => {
    for (const [r, obj] of Object.entries(py_testcases)) {
        it(`Should decode "${r}"`, () => {
            // hack to get around "TypeError: Cannot read properties of null (reading 'should')"
            expect(rison.decode(r)).toEqual(obj);
        });
        it(`Should encode ${JSON.stringify(obj)}`, () => {
            expect(rison.encode(obj)).toEqual(r);
        });
    }
    for (const [obj, r] of Object.entries(encode_testcases)) {
        it(`Should encode ${JSON.stringify(obj)}`, () => {
            expect(rison.encode(obj)).toEqual(r);
        });
    }
});
