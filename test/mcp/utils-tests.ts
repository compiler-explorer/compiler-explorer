// Copyright (C) 2026 Hudson River Trading LLC <opensource@hudson-trading.com>
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

import {applyCap, applyMatch, truncateLines} from '../../lib/mcp/utils.js';

type Item = {id: string; name: string};

const items: Item[] = [
    {id: 'g142', name: 'GCC 14.2'},
    {id: 'g141', name: 'GCC 14.1'},
    {id: 'clang20', name: 'Clang 20'},
    {id: 'msvc1944', name: 'MSVC 19.44'},
];

describe('applyMatch', () => {
    const extract = (i: Item) => [i.id, i.name];

    it('returns the input unchanged when pattern is undefined', () => {
        expect(applyMatch(items, undefined, extract)).toBe(items);
    });

    it('returns the input unchanged when pattern is empty', () => {
        expect(applyMatch(items, '', extract)).toBe(items);
    });

    it('matches against id substring case-insensitively', () => {
        expect(applyMatch(items, 'CLANG', extract)).toEqual([{id: 'clang20', name: 'Clang 20'}]);
    });

    it('matches against name substring', () => {
        expect(applyMatch(items, 'gcc', extract)).toHaveLength(2);
    });

    it('returns empty when nothing matches', () => {
        expect(applyMatch(items, 'rust', extract)).toEqual([]);
    });

    it('treats punctuation as whitespace so parens do not break matches', () => {
        const compilers = [
            {id: 'gcc-trunk-x64', name: 'x86-64 gcc (trunk)'},
            {id: 'gcc141', name: 'x86-64 gcc 14.1'},
        ];
        expect(applyMatch(compilers, 'x86-64 gcc trunk', i => [i.id, i.name])).toEqual([
            {id: 'gcc-trunk-x64', name: 'x86-64 gcc (trunk)'},
        ]);
    });

    it('matches tokens in any order', () => {
        const compilers = [
            {id: 'gcc141-arm64', name: 'arm64 gcc 14.1'},
            {id: 'gcc141', name: 'x86-64 gcc 14.1'},
        ];
        expect(applyMatch(compilers, 'gcc arm64 14.1', i => [i.id, i.name])).toEqual([
            {id: 'gcc141-arm64', name: 'arm64 gcc 14.1'},
        ]);
    });

    it('requires every token to be present', () => {
        const compilers = [{id: 'gcc141', name: 'x86-64 gcc 14.1'}];
        expect(applyMatch(compilers, 'gcc rust', i => [i.id, i.name])).toEqual([]);
    });

    it('treats a punctuation-only pattern as no filter', () => {
        expect(applyMatch(items, '(), .', extract)).toBe(items);
    });

    it('numeric tokens match whole-word so 14.1 does not match 14.10', () => {
        const compilers = [
            {id: 'g141', name: 'GCC 14.1'},
            {id: 'g1410', name: 'GCC 14.10'},
        ];
        expect(applyMatch(compilers, 'gcc 14.1', i => [i.id, i.name])).toEqual([{id: 'g141', name: 'GCC 14.1'}]);
    });

    it('numeric tokens still match the right whole word', () => {
        const compilers = [
            {id: 'g142', name: 'GCC 14.2'},
            {id: 'g141', name: 'GCC 14.1'},
            {id: 'g152', name: 'GCC 15.2'},
        ];
        expect(
            applyMatch(compilers, 'gcc 14', i => [i.id, i.name])
                .map(c => c.id)
                .sort(),
        ).toEqual(['g141', 'g142']);
    });

    it('alphanumeric tokens still match as substrings (so g14 finds g142)', () => {
        const compilers = [
            {id: 'g142', name: 'GCC 14.2'},
            {id: 'g141', name: 'GCC 14.1'},
            {id: 'clang20', name: 'Clang 20'},
        ];
        expect(
            applyMatch(compilers, 'g14', i => [i.id, i.name])
                .map(c => c.id)
                .sort(),
        ).toEqual(['g141', 'g142']);
    });
});

describe('applyCap', () => {
    type Compiler = {id: string; name: string; lang: string; version: string};
    const compilers: Compiler[] = [
        {id: 'g142', name: 'GCC 14.2', lang: 'c++', version: '14.2.0'},
        {id: 'g141', name: 'GCC 14.1', lang: 'c++', version: '14.1.0'},
        {id: 'clang20', name: 'Clang 20', lang: 'c++', version: '20.0.0'},
    ];
    const fullMap = (c: Compiler) => ({id: c.id, name: c.name, lang: c.lang, version: c.version});

    it('returns full-mapped items when count is under cap', () => {
        const result = applyCap(compilers, 10, fullMap, 'compilers');
        expect(result.items).toHaveLength(3);
        expect(result.items[0]).toEqual({id: 'g142', name: 'GCC 14.2', lang: 'c++', version: '14.2.0'});
        expect(result.total).toBe(3);
        expect(result.leanMode).toBeUndefined();
        expect(result.hint).toBeUndefined();
    });

    it('returns full-mapped items when count exactly matches cap', () => {
        const result = applyCap(compilers, 3, fullMap, 'compilers');
        expect(result.items).toHaveLength(3);
        expect(result.leanMode).toBeUndefined();
    });

    it('degrades to default lean shape (id+name) when count exceeds cap', () => {
        const result = applyCap(compilers, 1, fullMap, 'compilers');
        expect(result.items).toHaveLength(3);
        expect(result.items[0]).toEqual({id: 'g142', name: 'GCC 14.2'});
        expect(result.total).toBe(3);
        expect(result.leanMode).toBe(true);
        expect(result.hint).toMatch(/3 compilers/);
        expect(result.hint).toMatch(/cap of 1/);
    });

    it('uses a custom leanMap when provided', () => {
        const result = applyCap(
            compilers,
            1,
            fullMap,
            'compilers',
            c => ({onlyId: c.id}) as unknown as {id: string; name: string},
        );
        expect(result.items).toHaveLength(3);
        expect(result.items[0]).toEqual({onlyId: 'g142'});
        expect(result.leanMode).toBe(true);
    });

    it('handles missing name in default lean shape', () => {
        const items = [{id: 'libfoo'}, {id: 'libbar'}];
        const result = applyCap(items, 1, x => x, 'libraries');
        expect(result.items[0]).toEqual({id: 'libfoo', name: ''});
    });

    it('forceLean returns lean shape regardless of count, with no hint', () => {
        const result = applyCap(compilers, 100, fullMap, 'compilers', undefined, true);
        expect(result.items).toHaveLength(3);
        expect(result.items[0]).toEqual({id: 'g142', name: 'GCC 14.2'});
        expect(result.leanMode).toBe(true);
        expect(result.hint).toBeUndefined();
    });

    describe('lean hard cap', () => {
        const big = Array.from({length: 250}, (_, i) => ({id: `c${i}`, name: `Compiler ${i}`}));

        it('caps lean response at LEAN_HARD_CAP (200) when total exceeds it', () => {
            // Auto-degrade path: small maxResults forces lean, big input triggers hard cap.
            const result = applyCap(big, 25, x => x, 'compilers');
            expect(result.items).toHaveLength(200);
            expect(result.total).toBe(250);
            expect(result.leanMode).toBe(true);
            expect(result.hint).toMatch(/250 compilers matched/);
            expect(result.hint).toMatch(/first 200/);
            expect(result.hint).toMatch(/Refine with/);
        });

        it('caps lean response at LEAN_HARD_CAP when forceLean is set', () => {
            const result = applyCap(big, 1000, x => x, 'compilers', undefined, true);
            expect(result.items).toHaveLength(200);
            expect(result.total).toBe(250);
            expect(result.leanMode).toBe(true);
            expect(result.hint).toMatch(/Lean response capped at 200 of 250/);
        });

        it('does not cap when total is at or below the hard cap', () => {
            const small = Array.from({length: 200}, (_, i) => ({id: `c${i}`, name: `C${i}`}));
            const result = applyCap(small, 25, x => x, 'compilers');
            expect(result.items).toHaveLength(200);
            expect(result.leanMode).toBe(true);
            // Hint mentions the full-cap degradation but not truncation.
            expect(result.hint).toMatch(/200 compilers exceeded the full-detail cap/);
            expect(result.hint).not.toMatch(/first 200/);
        });
    });
});

describe('truncateLines', () => {
    const lines = [{text: 'a'}, {text: 'b'}, {text: 'c'}, {text: 'd'}];

    it('handles null input', () => {
        expect(truncateLines(null, 10)).toEqual({text: '', truncated: false, totalLines: 0});
    });

    it('handles undefined input', () => {
        expect(truncateLines(undefined, 10)).toEqual({text: '', truncated: false, totalLines: 0});
    });

    it('joins lines under cap with newlines', () => {
        expect(truncateLines(lines, 10)).toEqual({text: 'a\nb\nc\nd', truncated: false, totalLines: 4});
    });

    it('truncates and reports total when over cap', () => {
        expect(truncateLines(lines, 2)).toEqual({text: 'a\nb', truncated: true, totalLines: 4});
    });

    it('does not flag truncation at exact cap', () => {
        expect(truncateLines(lines, 4)).toEqual({text: 'a\nb\nc\nd', truncated: false, totalLines: 4});
    });

    it('strips ANSI colour codes from line text', () => {
        // Real gcc diagnostic shape captured from a staging compile — colour escapes
        // make the output noisy for an LLM consumer that doesn't have a terminal.
        const ESC = '\x1B';
        const ansiLines = [
            {text: `${ESC}[01m${ESC}[K<source>:${ESC}[m${ESC}[K In function`},
            {text: `${ESC}[01;31m${ESC}[Kerror: ${ESC}[m${ESC}[K'foo' was not declared`},
        ];
        const result = truncateLines(ansiLines, 10);
        expect(result.text).toBe("<source>: In function\nerror: 'foo' was not declared");
        expect(result.text).not.toContain(ESC);
    });
});
