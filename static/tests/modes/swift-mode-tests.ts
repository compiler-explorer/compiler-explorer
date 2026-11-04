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

import '../../modes/swift-mode.js';

/** Each line's tokens as [text, type], with the language suffix and whitespace-only tokens dropped. */
function tokens(source: string): [string, string][][] {
    const lines = source.split('\n');
    return monaco.editor
        .tokenize(source, 'swiftp')
        .map((row, index) =>
            row
                .map((token, i): [string, string] => [
                    lines[index].slice(token.offset, row[i + 1]?.offset),
                    token.type.replace(/\.swift$/, ''),
                ])
                .filter(([text]) => text.trim() !== ''),
        );
}

describe('swift mode', () => {
    it('reads an empty block comment as a complete comment', () => {
        const rows = tokens(['/**/ let x = 1', 'print(x)'].join('\n'));
        expect(rows[0]).toEqual([
            ['/**/', 'comment'],
            ['let', 'keyword'],
            ['x', 'identifier'],
            ['=', 'operator'],
            ['1', 'number'],
        ]);
        expect(rows[1][0]).toEqual(['print', 'identifier']);
    });

    it('still reads a doc comment through to its end', () => {
        const rows = tokens(['/** :param: x */ let', '/**', ' * docs', ' */', 'let'].join('\n'));
        expect(rows[0]).toEqual([
            ['/** ', 'comment.doc'],
            [':param:', 'comment.doc.param'],
            [' x */', 'comment.doc'],
            ['let', 'keyword'],
        ]);
        expect(rows[2]).toEqual([[' * docs', 'comment.doc']]);
        expect(rows[4]).toEqual([['let', 'keyword']]);
    });

    it('still reads nested block comments', () => {
        const rows = tokens('/* a /**/ b /* c */ d */ let');
        expect(rows[0]).toEqual([
            ['/* a /**/ b /* c */ d */', 'comment'],
            ['let', 'keyword'],
        ]);
    });
});
