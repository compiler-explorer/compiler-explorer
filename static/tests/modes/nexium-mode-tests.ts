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

import * as monaco from 'monaco-editor';
import {describe, expect, it} from 'vitest';

import '../../modes/nexium-mode.js';

function tokens(source: string): [string, string][] {
    const row = monaco.editor.tokenize(source, 'nexium')[0];
    return row
        .map((token, i): [string, string] => [
            source.slice(token.offset, row[i + 1]?.offset),
            token.type.replace(/\.nexium$/, ''),
        ])
        .filter(([text]) => text.trim() !== '');
}

describe('nexium mode', () => {
    it('tokenizes an anonymous literal without throwing', () => {
        expect(tokens('println("{}", .{square(12)})')).toEqual([
            ['println', 'identifier'],
            ['(', 'delimiter.parenthesis'],
            ['"', 'string'],
            ['{}', 'variable'],
            ['"', 'string'],
            [',', 'delimiter'],
            ['.{', 'delimiter.curly'],
            ['square', 'identifier'],
            ['(', 'delimiter.parenthesis'],
            ['12', 'number'],
            [')', 'delimiter.parenthesis'],
            ['}', 'delimiter.curly'],
            [')', 'delimiter.parenthesis'],
        ]);
    });

    it('tokenizes a binary pattern without throwing', () => {
        const brackets = tokens('<<a, b>>').filter(([, type]) => type.startsWith('delimiter.binary'));
        expect(brackets.map(([text]) => text)).toEqual(['<<', '>>']);
    });

    it('keeps compound shift assignments as single operators', () => {
        expect(tokens('x <<= 2')).toContainEqual(['<<=', 'operator']);
        expect(tokens('x >>= 2')).toContainEqual(['>>=', 'operator']);
    });
});
