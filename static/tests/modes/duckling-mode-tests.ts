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

import '../../modes/duckling-mode.js';

function tokens(source: string): [string, string][][] {
    const lines = source.split('\n');
    return monaco.editor
        .tokenize(source, 'duckling')
        .map((row, index) =>
            row
                .map((token, i): [string, string] => [
                    lines[index].slice(token.offset, row[i + 1]?.offset),
                    token.type.replace(/\.duckling$/, ''),
                ])
                .filter(([text]) => text.trim() !== ''),
        );
}

describe('Duckling keywords and types', () => {
    it('highlights pointer and object keywords', () => {
        expect(tokens('return ptrof self;')[0]).toEqual([
            ['return', 'keyword'],
            ['ptrof', 'keyword'],
            ['self', 'keyword'],
            [';', 'delimiter'],
        ]);
    });

    it.each(['i64', 'str', 'type', 'void', 'Array', 'Dict', 'Set'])('highlights %s as a builtin return type', type => {
        expect(tokens(`fundecl describe() -> ${type}`)[0]).toEqual([
            ['fundecl', 'keyword'],
            ['describe', 'identifier'],
            ['()', 'delimiter.parenthesis'],
            ['->', 'operator'],
            [type, 'type.identifier'],
        ]);
    });
});

describe('Duckling numeric literals', () => {
    it.each([
        ['42i32', 'number'],
        ['0xFFu64', 'number.hex'],
        ['0B1010u8', 'number.binary'],
        ['0o755u16', 'number.octal'],
        ['.5e+2f32', 'number'],
        ['1.e-3f64', 'number'],
        ['1.', 'number'],
    ])('highlights %s as one numeric literal', (literal, type) => {
        expect(tokens(literal)[0]).toEqual([[literal, type]]);
    });

    it('ends base-prefixed numbers before underscores', () => {
        expect(tokens('0xFF_00 0b10_01')[0]).toEqual([
            ['0xFF', 'number.hex'],
            ['_00', 'identifier'],
            ['0b10', 'number.binary'],
            ['_01', 'identifier'],
        ]);
    });
});

describe('Duckling comments', () => {
    it('keeps block comments across lines and resumes after the closing delimiter', () => {
        expect(tokens('#{ comment\nreturn 1;\n#} let')).toEqual([
            [['#{ comment', 'comment']],
            [['return 1;', 'comment']],
            [
                ['#}', 'comment'],
                ['let', 'keyword'],
            ],
        ]);
    });

    it('closes an empty block comment before following code', () => {
        expect(tokens('#{#} let')[0]).toEqual([
            ['#{#}', 'comment'],
            ['let', 'keyword'],
        ]);
    });
});
