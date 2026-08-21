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

import $ from 'jquery';
import {beforeEach, describe, expect, it, vi} from 'vitest';

import {CompilerOverrideType} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {CompilerInfo} from '../../types/compiler.interfaces.js';
import {CompilerShared} from '../compiler-shared.js';
import {compilersService} from '../services/compilers.service.js';
import {deserialiseState} from '../url.js';

vi.mock('../services/compilers.service.js', () => ({
    compilersService: {
        getCompilersForLang: vi.fn(),
    },
}));

// Fragment of a link whose overrides (edition, arch) were dropped on load.
// See PR#9047
// Reduced to the row -> stack -> compiler-component skeleton with only the componentState fields the widget path reads.
const urlHash =
    "g:!((g:!((g:!((h:compiler,i:(compiler:r1970,lang:rust,overrides:!((name:edition,value:'2021'),(name:arch,value:aarch64-unknown-linux-musl))),l:'5')),l:'4')),l:'2')),version:4";

const rustc1970 = {
    id: 'r1970',
    lang: 'rust',
    possibleOverrides: [
        {
            type: 'options',
            name: CompilerOverrideType.edition,
            display_title: 'Edition',
            description: '',
            flags: ['--edition', '<value>'],
            values: [
                {name: '2015', value: '2015'},
                {name: '2018', value: '2018'},
                {name: '2021', value: '2021'},
                {name: '2024', value: '2024'},
            ],
            default: '2024',
        },
        {
            type: 'options',
            name: CompilerOverrideType.arch,
            display_title: 'Architecture',
            description: '',
            flags: ['--target=<value>'],
            values: [
                {name: 'x86_64-unknown-linux-gnu', value: 'x86_64-unknown-linux-gnu'},
                {name: 'aarch64-unknown-linux-musl', value: 'aarch64-unknown-linux-musl'},
            ],
        },
    ],
} as unknown as CompilerInfo;

const gcc152 = {
    id: 'g152',
    lang: 'c++',
    possibleOverrides: [
        {
            type: 'options',
            name: CompilerOverrideType.stdver,
            display_title: 'Std version',
            description: '',
            flags: ['-std=<value>'],
            values: [{name: 'c++17', value: 'c++17'}],
        },
    ],
} as unknown as CompilerInfo;

const urlOverrides = [
    {name: 'edition', value: '2021'},
    {name: 'arch', value: 'aarch64-unknown-linux-musl'},
];

function findComponent(content: any[], name: string): any {
    for (const item of content) {
        if (item.componentName === name) {
            return item;
        }

        if (item.content && Array.isArray(item.content)) {
            const found = findComponent(item.content, name);

            if (found) {
                return found;
            }
        }
    }
    return null;
}

function makeDomRoot(): JQuery {
    return $('<div><button class="btn show-overrides"></button></div>');
}

describe('CompilerShared override restoration', () => {
    beforeEach(() => {
        vi.mocked(compilersService.getCompilersForLang).mockReset();
    });

    it('deserialises the overrides from the URL fragment', () => {
        const state = deserialiseState(urlHash);
        const compiler = findComponent(state.content, 'compiler');

        expect(compiler.componentState.compiler).toBe('r1970');
        expect(compiler.componentState.lang).toBe('rust');
        expect(compiler.componentState.overrides).toEqual(urlOverrides);
    });

    it('applies the URL overrides instead of the compiler defaults', async () => {
        vi.mocked(compilersService.getCompilersForLang).mockResolvedValue({r1970: rustc1970});

        const state = deserialiseState(urlHash);
        const componentState = findComponent(state.content, 'compiler').componentState;
        const shared = new CompilerShared(makeDomRoot(), () => {});

        await shared.updateState(componentState);

        expect(shared.getOverrides()).toEqual(urlOverrides);
    });

    it('keeps the URL overrides when the pane echoes its state back', async () => {
        vi.mocked(compilersService.getCompilersForLang).mockResolvedValue({r1970: rustc1970});

        const state = deserialiseState(urlHash);
        const componentState = findComponent(state.content, 'compiler').componentState;
        const shared = new CompilerShared(makeDomRoot(), () => {});

        await shared.updateState(componentState);

        await shared.updateState({...componentState, overrides: shared.getOverrides()});

        expect(shared.getOverrides()).toEqual(urlOverrides);
    });

    it('ignores a stale compiler-list fetch that resolves after a newer call', async () => {
        let resolveRust!: (value: Record<string, CompilerInfo>) => void;

        const rustPromise = new Promise<Record<string, CompilerInfo>>(resolve => {
            resolveRust = resolve;
        });

        vi.mocked(compilersService.getCompilersForLang).mockImplementation(lang =>
            lang === 'rust' ? rustPromise : Promise.resolve({g152: gcc152}),
        );

        const shared = new CompilerShared(makeDomRoot(), () => {});

        const stale = shared.updateState({
            compiler: 'r1970',
            lang: 'rust',
            overrides: [{name: 'edition', value: '2018'}],
        } as any);

        const fresh = shared.updateState({
            compiler: 'g152',
            lang: 'c++',
            overrides: [{name: 'stdver', value: 'c++17'}],
        } as any);

        await fresh;

        resolveRust({r1970: rustc1970});

        await stale;

        expect(shared.getOverrides()).toEqual([{name: 'stdver', value: 'c++17'}]);
    });
});
