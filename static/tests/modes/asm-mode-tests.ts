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

import '../../modes/asm-mode.js';

/** Each line's tokens as [text, type], with the language suffix and whitespace-only tokens dropped. */
function tokens(source: string): [string, string][][] {
    const lines = source.split('\n');
    return monaco.editor
        .tokenize(source, 'asm')
        .map((row, index) =>
            row
                .map((token, i): [string, string] => [
                    lines[index].slice(token.offset, row[i + 1]?.offset),
                    token.type.replace(/\.asm$/, ''),
                ])
                .filter(([text]) => text.trim() !== ''),
        );
}

describe('asm mode', () => {
    it('does not open a string at a lone backtick in a SASS label reference', () => {
        const rows = tokens(
            [' EXIT ', '.L_x_0:', ' BRA `(.L_x_0) ', '.L_x_2:', 'add(int*, int):', ' MOV R1, c[0x0][0x20] '].join('\n'),
        );
        expect(rows[2]).toEqual([
            ['BRA', 'keyword'],
            ['`', 'operator'],
            ['(', 'delimiter.parenthesis'],
            ['.L_x_0', 'type.identifier'],
            [')', 'delimiter.parenthesis'],
        ]);
        expect(rows[3]).toEqual([['.L_x_2:', 'type.identifier']]);
        expect(rows[4]).toEqual([['add(int*, int):', 'type.identifier']]);
        expect(rows[5][0]).toEqual(['MOV', 'keyword']);
    });

    it('still reads an MSVC backtick string after an opcode', () => {
        const rows = tokens(["\tlea\trcx, OFFSET FLAT:`string'", '\tret\t0'].join('\n'));
        expect(rows[0].slice(-3)).toEqual([
            ['`', 'string.backtick'],
            ['string', 'string'],
            ["'", 'string.backtick'],
        ]);
        expect(rows[1][0]).toEqual(['ret', 'keyword']);
    });

    it('still reads an MSVC backtick string at the start of a line', () => {
        const rows = tokens(["`string', 00H", '\tret\t0'].join('\n'));
        expect(rows[0].slice(0, 3)).toEqual([
            ['`', 'string.backtick'],
            ['string', 'string'],
            ["'", 'string.backtick'],
        ]);
        expect(rows[1][0]).toEqual(['ret', 'keyword']);
    });

    it('does not let an unterminated backtick at the start of a line swallow the next line', () => {
        const rows = tokens(['`(.L_x_0)', '\tret\t0'].join('\n'));
        expect(rows[0].some(([, type]) => type.startsWith('string'))).toBe(false);
        expect(rows[1][0]).toEqual(['ret', 'keyword']);
    });
});
