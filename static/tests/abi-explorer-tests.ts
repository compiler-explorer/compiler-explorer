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

import {decode} from '@ambrus-toth/abi-explorer-share-link';
import {describe, expect, it} from 'vitest';

import {
    ABI_EXPLORER_URL,
    abiExplorerLanguageFor,
    abiExplorerState,
    abiExplorerTripleFor,
    getAbiExplorerUrl,
    getAbiExplorerUrlSync,
    normalizeAbiExplorerStd,
} from '../abi-explorer.js';

describe('ABI Explorer links', () => {
    describe('abiExplorerLanguageFor', () => {
        it('accepts the languages ABI Explorer lays out', () => {
            expect(abiExplorerLanguageFor('c')).toBe('c');
            expect(abiExplorerLanguageFor('c++')).toBe('c++');
            expect(abiExplorerLanguageFor('hylo')).toBe('hylo');
        });

        it('rejects the rest', () => {
            expect(abiExplorerLanguageFor('rust')).toBeNull();
            expect(abiExplorerLanguageFor(undefined)).toBeNull();
        });
    });

    describe('normalizeAbiExplorerStd', () => {
        it('strips the flag and folds draft names onto released ones', () => {
            expect(normalizeAbiExplorerStd('-std=gnu++2b')).toBe('gnu++23');
            expect(normalizeAbiExplorerStd('c++2a')).toBe('c++20');
            expect(normalizeAbiExplorerStd('C18')).toBe('c17');
        });

        it('passes other values through for ABI Explorer to judge', () => {
            expect(normalizeAbiExplorerStd('c++20')).toBe('c++20');
            expect(normalizeAbiExplorerStd('c++29')).toBe('c++29');
        });
    });

    describe('abiExplorerTripleFor', () => {
        it('prefers the configured target over the banner', () => {
            expect(
                abiExplorerTripleFor('aarch64', 'clang\nTarget: x86_64-unknown-linux-gnu', '-target aarch64-linux-gnu'),
            ).toBe('aarch64-linux-gnu');
            expect(abiExplorerTripleFor('arm32', '', '--target=arm-linux-gnueabi -O2')).toBe('arm-linux-gnueabi');
        });

        it('uses the banner when nothing is configured', () => {
            expect(abiExplorerTripleFor('amd64', 'clang version 20\nTarget: x86_64-w64-windows-gnu\n')).toBe(
                'x86_64-w64-windows-gnu',
            );
        });

        it('ignores a candidate that disagrees with the instruction set', () => {
            expect(abiExplorerTripleFor('aarch64', 'Target: x86_64-unknown-linux-gnu')).toBe(
                'aarch64-unknown-linux-gnu',
            );
        });

        it('passes through targets for instruction sets it does not know', () => {
            expect(abiExplorerTripleFor('avr', '', '-target avr')).toBe('avr');
        });

        it('names nothing where nothing is known', () => {
            expect(abiExplorerTripleFor('mips')).toBeUndefined();
            expect(abiExplorerTripleFor('avr')).toBeUndefined();
            expect(abiExplorerTripleFor(null)).toBeUndefined();
        });
    });

    describe('abiExplorerState', () => {
        it('opens on defaults with no compiler', () => {
            const state = abiExplorerState('struct S {};', 'c++', []);
            expect(state.buffers).toHaveLength(1);
            expect(state.buffers[0].options).toMatchObject({lang: 'c++', std: '', triple: ''});
        });

        it('makes a buffer per distinct standard and target', () => {
            const state = abiExplorerState('struct S {};', 'c++', [
                {name: 'x86-64 clang', std: 'c++20', triple: 'x86_64-unknown-linux-gnu'},
                {name: 'x86-64 clang (other)', std: 'c++20', triple: 'x86_64-unknown-linux-gnu'},
                {name: 'armv8 clang', std: 'c++20', triple: 'aarch64-linux-gnu'},
            ]);
            expect(state.buffers.map(b => b.name)).toEqual(['x86-64 clang', 'armv8 clang']);
            expect(state.buffers.every(b => b.source === 'struct S {};')).toBe(true);
        });

        it('carries no more buffers than a link may', () => {
            const compilers = Array.from({length: 12}, (_, i) => ({name: `c${i}`, std: '', triple: `t${i}`}));
            expect(abiExplorerState('', 'c', compilers).buffers).toHaveLength(8);
        });
    });

    describe('encoding', () => {
        const state = abiExplorerState('struct S { char c; int i; };', 'c++', [
            {name: 'armv8 clang', std: 'c++23', triple: 'aarch64-linux-gnu'},
        ]);

        it.each([
            ['compressed', getAbiExplorerUrl],
            ['uncompressed', async (s: typeof state) => getAbiExplorerUrlSync(s)],
        ])('round-trips the %s link through ABI Explorer’s reader', async (_name, getUrl) => {
            const url = await getUrl(state);
            expect(url.startsWith(`${ABI_EXPLORER_URL}#`)).toBe(true);
            expect(await decode(url.slice(ABI_EXPLORER_URL.length))).toEqual(state);
        });
    });
});
