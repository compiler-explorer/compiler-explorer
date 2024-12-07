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

import {addDigitSeparator, escapeHTML, splitArguments} from '../shared/common-utils.js';

describe('HTML Escape Test Cases', () => {
    it('should prevent basic injection', () => {
        expect(escapeHTML("<script>alert('hi');</script>")).toEqual(
            `&lt;script&gt;alert(&#x27;hi&#x27;);&lt;/script&gt;`,
        );
    });
    it('should prevent tag injection', () => {
        expect(escapeHTML('\'"`>')).toEqual(`&#x27;&quot;&#x60;&gt;`);
    });
});

describe('digit separator', () => {
    it('handles short numbers', () => {
        expect(addDigitSeparator('42', '_', 3)).toEqual('42');
    });
    it('handles long numbers', () => {
        expect(addDigitSeparator('1234', '_', 3)).toEqual('1_234');
        expect(addDigitSeparator('123456789', "'", 3)).toEqual("123'456'789");
        expect(addDigitSeparator('1234567890', "'", 3)).toEqual("1'234'567'890");
    });
    it('handles hex numbers', () => {
        expect(addDigitSeparator('AABBCCDD12345678', '_', 4)).toEqual('AABB_CCDD_1234_5678');
        expect(addDigitSeparator('01AABBCCDD12345678', '_', 4)).toEqual('01_AABB_CCDD_1234_5678');
    });
    it('handles negative numbers', () => {
        expect(addDigitSeparator('-42', '_', 3)).toEqual('-42');
        expect(addDigitSeparator('-420', '_', 3)).toEqual('-420');
        expect(addDigitSeparator('-4200', '_', 3)).toEqual('-4_200');
        expect(addDigitSeparator('-123456789', '_', 3)).toEqual('-123_456_789');
    });
});

describe('argument splitting', () => {
    it('should handle empty strings', () => {
        expect(splitArguments('')).toEqual([]);
    });
    it('should handle purely whitespace strings', () => {
        expect(splitArguments('    ')).toEqual([]);
    });
    it('should handle normal things', () => {
        expect(splitArguments('-hello --world etc --std=c++20')).toEqual(['-hello', '--world', 'etc', '--std=c++20']);
    });

    it('should handle hash chars', () => {
        expect(splitArguments('-Wno#warnings -Wno-#pragma-messages')).toEqual([
            '-Wno#warnings',
            '-Wno-#pragma-messages',
        ]);
    });

    it('should handle doublequoted args', () => {
        expect(splitArguments('--hello "-world etc"')).toEqual(['--hello', '-world etc']);
    });

    it('should handle singlequoted args', () => {
        expect(splitArguments("--hello '-world etc'")).toEqual(['--hello', '-world etc']);
    });

    it('should handle mismatched doublequoted args', () => {
        expect(splitArguments('--hello "-world etc')).toEqual(['--hello', '-world etc']);
    });

    it('should handle mismatched singlequoted args', () => {
        expect(splitArguments("--hello '-world etc")).toEqual(['--hello', '-world etc']);
    });

    it('should handle hashes at the beginning of words', () => {
        expect(splitArguments('hello #veryfancy etc')).toEqual(['hello', '#veryfancy', 'etc']);
    });

    it('should handle empty arguments', () => {
        expect(splitArguments('nothing "" is lost \'\'')).toEqual(['nothing', '', 'is', 'lost', '']);
    });

    it('should handle weird characters', () => {
        expect(splitArguments('hello | moose | \\n < yay & \n\t\nmoo')).toEqual([
            'hello',
            '|',
            'moose',
            '|',
            'n',
            '<',
            'yay',
            '&',
            'moo',
        ]);
    });

    it('should handle escaped backslashes', () => {
        expect(splitArguments('hello \\\\ moose')).toEqual(['hello', '\\', 'moose']);
    });

    it('should handle mismatched backslashes', () => {
        expect(splitArguments('hello \\')).toEqual(['hello', '\\']);
        expect(splitArguments('hello world\\')).toEqual(['hello', 'world\\']);
    });
    it('should handle mismatched backslashes in quotes', () => {
        expect(splitArguments('hello "\\')).toEqual(['hello', '\\']);
    });

    it('should handle escaped quotes', () => {
        expect(splitArguments('hi \\"mum\\"')).toEqual(['hi', '"mum"']);
        expect(splitArguments('hi "\\"mum\\""')).toEqual(['hi', '"mum"']);
        expect(splitArguments("hi \\'mum\\'")).toEqual(['hi', "'mum'"]);
    });

    it('should do something sane in the face of escaped quotes in quotes', () => {
        // Not really valid input, bash sees this as a bunch of abutting single-quotes, unterminated.
        // But this makes as much sense as anything and so I've added it to lock in the behaviour.
        expect(splitArguments("hi '\\'mum\\''")).toEqual(['hi', "\\mum'"]);
    });

    it('should handle abutting quotes', () => {
        expect(splitArguments('well "hello""there"')).toEqual(['well', 'hellothere']);
        expect(splitArguments('well "hello"\'there\'')).toEqual(['well', 'hellothere']);
    });

    it('should ignore escaped hashes etc', () => {
        expect(splitArguments('hello \\#veryfancy etc')).toEqual(['hello', '#veryfancy', 'etc']);
    });

    it('should handle multiple spaces', () => {
        expect(splitArguments('hello    world')).toEqual(['hello', 'world']);
    });

    it('should handle escaped spaces', () => {
        expect(splitArguments('hello\\ there')).toEqual(['hello there']);
    });

    it('should handle mixed quotes and spaces', () => {
        expect(splitArguments('"hello \'world\' there"')).toEqual(["hello 'world' there"]);
    });

    it('should handle quotes containing spaces', () => {
        expect(splitArguments('hello "   " world')).toEqual(['hello', '   ', 'world']);
    });

    it('should handle multiple escapes in quotes', () => {
        expect(splitArguments('"hello \\"world\\" \\\\"')).toEqual(['hello "world" \\']);
    });
});
