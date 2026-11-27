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

import '../../modes/mach-ir-mode.js';
import '../../modes/mach-mode.js';

/** Each line's tokens as [text, type], with the language suffix and whitespace-only tokens dropped. */
function tokens(source: string, lang: 'mach' | 'mach-ir' = 'mach'): [string, string][][] {
    const suffix = new RegExp(`\\.${lang}$`);
    const lines = source.split('\n');
    return monaco.editor
        .tokenize(source, lang)
        .map((row, index) =>
            row
                .map((token, i): [string, string] => [
                    lines[index].slice(token.offset, row[i + 1]?.offset),
                    token.type.replace(suffix, ''),
                ])
                .filter(([text]) => text.trim() !== ''),
        );
}

describe('mach mode', () => {
    it('keeps a braced aarch64 register list and // comment inside an ISA-selected asm block', () => {
        const rows = tokens(
            [
                'asm armv8-a+fp+simd {',
                '    ld1 {v0.16b}, [x1], 16 // }}',
                '    eor v0.16b, v0.16b, v1.16b',
                '}',
                'ret 0;',
            ].join('\n'),
        );
        expect(rows[0]).toEqual([
            ['asm', 'keyword'],
            ['armv8-a+fp+simd', 'type'],
            ['{', 'delimiter.curly'],
        ]);
        expect(rows[1]).toEqual([
            ['    ld1 {v0.16b}, [x1], 16 ', ''],
            ['// }}', 'comment'],
        ]);
        expect(rows[2]).toEqual([['    eor v0.16b, v0.16b, v1.16b', '']]);
        expect(rows[3]).toEqual([['}', 'delimiter.curly']]);
        expect(rows[4][0]).toEqual(['ret', 'keyword']);
    });

    it('reads an identifier in braces as a local under a comma-separated x86-64 selection', () => {
        const rows = tokens(['asm x86-64,avx2 {', '    mov rax, {a}', '}', 'ret 0;'].join('\n'));
        expect(rows[0]).toEqual([
            ['asm', 'keyword'],
            ['x86-64,avx2', 'type'],
            ['{', 'delimiter.curly'],
        ]);
        expect(rows[1]).toEqual([
            ['    mov rax, ', ''],
            ['{a}', 'variable'],
        ]);
        expect(rows[3][0]).toEqual(['ret', 'keyword']);
    });

    it('reads a local inside a braced group, and ignores braces in # and ;; comments', () => {
        const rows = tokens(['asm wasm32 {', '    {x {a}} # }}}', '    local.get 0 ;; }}}', '}', 'ret 0;'].join('\n'));
        expect(rows[1]).toEqual([
            ['    {x ', ''],
            ['{a}', 'variable'],
            ['} ', ''],
            ['# }}}', 'comment'],
        ]);
        expect(rows[2]).toEqual([
            ['    local.get 0 ', ''],
            [';; }}}', 'comment'],
        ]);
        expect(rows[4][0]).toEqual(['ret', 'keyword']);
    });

    it('highlights wide scalars, SIMD vectors, multiline strings, and v7 decorators', () => {
        const rows = tokens(
            [
                '#[constant_time("hard")]',
                '#[gpu.stage("compute")]',
                'pub fun kern(flag: u1, wide: u256, half: f16, lanes: f32x4) {',
                '    val msg = \\\\hello;',
                '}',
            ].join('\n'),
        );
        expect(rows[0]).toEqual([
            ['#[constant_time', 'annotation'],
            ['(', 'delimiter'],
            ['"hard"', 'string'],
            [')', 'delimiter'],
            [']', 'annotation'],
        ]);
        expect(rows[1]).toEqual([
            ['#[gpu.stage', 'annotation'],
            ['(', 'delimiter'],
            ['"compute"', 'string'],
            [')', 'delimiter'],
            [']', 'annotation'],
        ]);
        expect(rows[2]).toContainEqual(['u1', 'type']);
        expect(rows[2]).toContainEqual(['u256', 'type']);
        expect(rows[2]).toContainEqual(['f16', 'type']);
        expect(rows[2]).toContainEqual(['f32x4', 'type']);
        expect(rows[3]).toContainEqual(['\\\\hello;', 'string']);
    });
});

describe('mach-ir mode', () => {
    it('tokenizes MIRL headers, nested decorators, dotted opcodes, and block labels', () => {
        const rows = tokens(
            [
                'mirl "0.3.1"; # module header',
                '#[placed(fun(i64) i64, [{reg(0, 7), fixed, exact}])] pub fun example.mask(_0: ^i64) i64 {',
                '_0(_2: ^i64):',
                '    val _3: ptr = ?example.LANES;',
                '    ret.placed[places: [{reg(0, 0), fixed, exact}]](_2) at l13;',
                '}',
            ].join('\n'),
            'mach-ir',
        );
        expect(rows[0]).toEqual([
            ['mirl', 'keyword'],
            ['"0.3.1"', 'string'],
            [';', 'delimiter'],
            ['# module header', 'comment'],
        ]);
        // nested `[...]` inside `#[placed(...)]` must stay in decorator state until the outer `]`
        expect(rows[1]).toContainEqual(['pub', 'keyword']);
        expect(rows[1]).toContainEqual(['fun', 'keyword']);
        expect(rows[2][0]).toEqual(['_0', 'type.identifier']);
        expect(rows[3]).toContainEqual(['?example.LANES', 'variable.predefined']);
        expect(rows[4]).toContainEqual(['ret.placed', 'keyword.operator']);
        expect(rows[4]).toContainEqual(['at', 'keyword']);
    });
});
