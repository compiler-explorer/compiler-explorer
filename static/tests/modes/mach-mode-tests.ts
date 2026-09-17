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

import '../../modes/mach-mode.js';

/** Each line's tokens as [text, type], with the language suffix and whitespace-only tokens dropped. */
function tokens(source: string): [string, string][][] {
    const lines = source.split('\n');
    return monaco.editor.tokenize(source, 'mach').map((row, index) =>
        row
            .map((token, i): [string, string] => [
                lines[index].slice(token.offset, row[i + 1]?.offset),
                token.type.replace(/\.mach$/, ''),
            ])
            .filter(([text]) => text.trim() !== ''),
    );
}

describe('mach mode', () => {
    it('keeps a braced aarch64 register list inside the asm block', () => {
        const rows = tokens(
            [
                'asm aarch64 {',
                '    ld1 {v0.16b}, [x1], 16',
                '    eor v0.16b, v0.16b, v1.16b',
                '}',
                'ret 0;',
            ].join('\n'),
        );
        expect(rows[1]).toEqual([['    ld1 {v0.16b}, [x1], 16', '']]);
        // were the register list's `}` to close the block, this line would read as mach
        expect(rows[2]).toEqual([['    eor v0.16b, v0.16b, v1.16b', '']]);
        expect(rows[3]).toEqual([['}', 'delimiter.curly']]);
        expect(rows[4][0]).toEqual(['ret', 'keyword']);
    });

    it('still reads an identifier in braces as a local', () => {
        const rows = tokens(['asm x86_64 {', '    mov rax, {a}', '}', 'ret 0;'].join('\n'));
        expect(rows[1]).toEqual([
            ['    mov rax, ', ''],
            ['{a}', 'variable'],
        ]);
        expect(rows[3][0]).toEqual(['ret', 'keyword']);
    });

    it('reads a local inside a braced group, and ignores braces in a comment', () => {
        const rows = tokens(['asm x86_64 {', '    {x {a}} # }}}', '}', 'ret 0;'].join('\n'));
        expect(rows[1]).toEqual([
            ['    {x ', ''],
            ['{a}', 'variable'],
            ['} ', ''],
            ['# }}}', 'comment'],
        ]);
        expect(rows[3][0]).toEqual(['ret', 'keyword']);
    });
});
