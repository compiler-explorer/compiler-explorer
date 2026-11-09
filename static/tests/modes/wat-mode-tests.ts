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

import '../../modes/wat-mode.js';

/** Each line's tokens as [text, type], with the language suffix and whitespace-only tokens dropped. */
function tokens(source: string): [string, string][][] {
    const lines = source.split('\n');
    return monaco.editor
        .tokenize(source, 'wat')
        .map((row, index) =>
            row
                .map((token, i): [string, string] => [
                    lines[index].slice(token.offset, row[i + 1]?.offset),
                    token.type.replace(/\.wat$/, ''),
                ])
                .filter(([text]) => text.trim() !== ''),
        );
}

describe('wat mode', () => {
    it('tokenizes typed instructions as keywords and value types as types', () => {
        const rows = tokens(
            ['(func $add (param $a i32) (result i32)', '  local.get $a', '  i32.const 42', '  i32.add)'].join('\n'),
        );
        expect(rows[0]).toEqual([
            ['(', 'delimiter.parenthesis'],
            ['func', 'keyword'],
            ['$add', 'identifier'],
            ['(', 'delimiter.parenthesis'],
            ['param', 'keyword'],
            ['$a', 'identifier'],
            ['i32', 'type'],
            [')', 'delimiter.parenthesis'],
            ['(', 'delimiter.parenthesis'],
            ['result', 'keyword'],
            ['i32', 'type'],
            [')', 'delimiter.parenthesis'],
        ]);
        expect(rows[1]).toEqual([
            ['local.get', 'keyword'],
            ['$a', 'identifier'],
        ]);
        expect(rows[2]).toEqual([
            ['i32.const', 'keyword'],
            ['42', 'number'],
        ]);
        expect(rows[3][0]).toEqual(['i32.add', 'keyword']);
    });

    it('tokenizes identifiers containing any idchar', () => {
        expect(tokens('$a@b $foo^bar $std::rt::lang_start $a0')[0]).toEqual([
            ['$a@b', 'identifier'],
            ['$foo^bar', 'identifier'],
            ['$std::rt::lang_start', 'identifier'],
            ['$a0', 'identifier'],
        ]);
    });

    it('tokenizes integer and float literals', () => {
        const literals = ['-1', '0xFF_FF', '3.14e-2', '0x1p-3', '-inf', 'nan:0x7f'];
        expect(tokens(literals.join(' '))[0]).toEqual(literals.map(literal => [literal, 'number']));
    });

    it('does not tokenize a number glued to other idchars', () => {
        expect(tokens('1abc')[0]).toEqual([['1abc', '']]);
    });

    it('tokenizes line and nested block comments', () => {
        const rows = tokens(
            [';; line (; not a block', '(; outer (; inner ;) still outer ;) nop', '(; open', 'closed ;) nop'].join(
                '\n',
            ),
        );
        expect(rows[0]).toEqual([[';; line (; not a block', 'comment']]);
        expect(rows[1]).toEqual([
            ['(; outer (; inner ;) still outer ;)', 'comment'],
            ['nop', 'keyword'],
        ]);
        expect(rows[2]).toEqual([['(; open', 'comment']]);
        expect(rows[3]).toEqual([
            ['closed ;)', 'comment'],
            ['nop', 'keyword'],
        ]);
    });

    it('tokenizes string escapes', () => {
        expect(tokens('(data "a\\n\\u{41}\\ff\\q")')[0]).toEqual([
            ['(', 'delimiter.parenthesis'],
            ['data', 'keyword'],
            ['"a', 'string'],
            ['\\n\\u{41}\\ff', 'string.escape'],
            ['\\q', 'string.escape.invalid'],
            ['"', 'string'],
            [')', 'delimiter.parenthesis'],
        ]);
    });

    it('tokenizes annotations', () => {
        expect(tokens('(@custom "x")')[0]).toEqual([
            ['(', 'delimiter.parenthesis'],
            ['@custom', 'annotation'],
            ['"x"', 'string'],
            [')', 'delimiter.parenthesis'],
        ]);
    });

    it('does not throw on input that is not WAT', () => {
        expect(() =>
            tokens(["const _0x6d8f = ['copy_char', 'value'];", 'x = [1, 2]', '{}@^`~,'].join('\n')),
        ).not.toThrow();
    });
});
