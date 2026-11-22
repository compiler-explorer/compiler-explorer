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

import '../../modes/ruby-mode.js';

/** Each line's tokens as [text, type], with the language suffix and whitespace-only tokens dropped. */
function tokens(source: string): [string, string][][] {
    const lines = source.split('\n');
    return monaco.editor
        .tokenize(source, 'rubyp')
        .map((row, index) =>
            row
                .map((token, i): [string, string] => [
                    lines[index].slice(token.offset, row[i + 1]?.offset),
                    token.type.replace(/\.ruby$/, ''),
                ])
                .filter(([text]) => text.trim() !== ''),
        );
}

describe('ruby mode', () => {
    it('tokenizes a conditional assigned to an indexed variable', () => {
        const rows = tokens(['ENV["A"] = if cond?', '  "x"', 'else', '  "y"', 'end', 'puts ENV["A"]'].join('\n'));
        expect(rows[0]).toContainEqual(['if', 'keyword.if']);
        expect(rows[4]).toEqual([['end', 'keyword.if']]);
        expect(rows[5][0]).toEqual(['puts', 'identifier']);
    });

    it('closes an assigned if, and treats a stray end as a keyword', () => {
        const rows = tokens(['x = if a', 'else', 'end', 'end'].join('\n'));
        expect(rows[0]).toEqual([
            ['x', 'identifier'],
            [' = ', ''],
            ['if', 'keyword.if'],
            ['a', 'identifier'],
        ]);
        expect(rows[1]).toEqual([['else', 'keyword']]);
        expect(rows[2]).toEqual([['end', 'keyword.if']]);
        expect(rows[3]).toEqual([['end', 'keyword']]);
    });

    it('closes an assigned while loop', () => {
        const rows = tokens(['y = while a do', 'end'].join('\n'));
        expect(rows[0]).toContainEqual(['while', 'keyword.while']);
        expect(rows[0]).toContainEqual(['do', 'keyword']);
        expect(rows[1]).toEqual([['end', 'keyword.while']]);
    });

    it('opens blocks for conditionals after brackets and separators', () => {
        const rows = tokens(
            ['def f', '  foo(if a', '  end, if b', '  end)', '  x; unless c', '  end', 'end'].join('\n'),
        );
        expect(rows[1]).toEqual([
            ['foo', 'identifier'],
            ['(', 'delimiter.parenthesis'],
            ['if', 'keyword.if'],
            ['a', 'identifier'],
        ]);
        expect(rows[2]).toEqual([
            ['end', 'keyword.if'],
            [',', 'delimiter'],
            ['if', 'keyword.if'],
            ['b', 'identifier'],
        ]);
        expect(rows[3]).toEqual([
            ['end', 'keyword.if'],
            [')', 'delimiter.parenthesis'],
        ]);
        expect(rows[4]).toContainEqual(['unless', 'keyword.unless']);
        expect(rows[5]).toEqual([['end', 'keyword.unless']]);
        expect(rows[6]).toEqual([['end', 'keyword.def']]);
    });

    it('still reads trailing modifiers as modifiers', () => {
        const rows = tokens(['def f', '  return if x', '  x = y unless z', 'end'].join('\n'));
        expect(rows[1]).toContainEqual(['if', 'keyword.ifx']);
        expect(rows[2]).toContainEqual(['unless', 'keyword.unlessx']);
        expect(rows[3]).toEqual([['end', 'keyword.def']]);
    });
});
