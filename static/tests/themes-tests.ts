// Copyright (c) 2026, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//     Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
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

import GoldenLayout from 'golden-layout';
import $ from 'jquery';
import {afterEach, describe, expect, it, vi} from 'vitest';

import {Compiler} from '../panes/compiler.js';
import {Executor} from '../panes/executor.js';
import {SiteSettings} from '../settings.js';
import {Themer, themes} from '../themes.js';

function makeEventHub() {
    const emit = vi.fn();
    return {eventHub: {on: vi.fn(), emit} as unknown as GoldenLayout.EventEmitter, emit};
}

describe('Themer', () => {
    afterEach(() => vi.restoreAllMocks());

    it('announces the theme it applies', () => {
        const {eventHub, emit} = makeEventHub();

        new Themer(eventHub, {theme: 'onedark'} as SiteSettings);

        expect(emit).toHaveBeenCalledWith('themeChange', themes.onedark);
    });

    it.each([
        [true, themes.dark],
        [false, themes.default],
    ])('announces the system theme as the one it resolved to when dark is preferred is %s', (prefersDark, theme) => {
        vi.spyOn(window, 'matchMedia').mockReturnValue({matches: prefersDark} as MediaQueryList);
        const {eventHub, emit} = makeEventHub();

        new Themer(eventHub, {theme: 'system'} as SiteSettings);

        expect(emit).toHaveBeenCalledWith('themeChange', theme);
    });
});

describe.each([
    ['Compiler', Compiler.prototype],
    ['Executor', Executor.prototype],
])('%s pane on a theme change', (_name, prototype) => {
    afterEach(() => vi.restoreAllMocks());

    function makePane(supportsDarkBackgroundColours: boolean) {
        const pane = Object.create(prototype) as Compiler | Executor;
        const compileFromEditorSource = vi.fn();
        Object.assign(pane, {
            compiler: {supportsDarkBackgroundColours},
            darkBackgroundColours: false,
            deferCompiles: false,
            sourceTreeId: null,
            compileInfoLabel: $('<span>'),
            compileTimeLabel: $('<span>'),
            compilerShared: {getOverrides: () => [], getRuntimeTools: () => undefined},
            getEffectiveFilters: () => ({}),
            getActiveTools: () => [],
            getCurrentState: () => ({}),
            compileFromEditorSource,
        });
        const sentDarkBackgroundColours = () =>
            compileFromEditorSource.mock.calls.map(([options]) => options.compilerOptions.darkBackgroundColours);
        return {pane, sentDarkBackgroundColours};
    }

    it('recompiles for a compiler that colours for the background when light and dark swap', () => {
        const {pane, sentDarkBackgroundColours} = makePane(true);

        pane.onThemeChange(themes.dark);
        pane.onThemeChange(themes.onedark);
        pane.onThemeChange(themes.pink);

        expect(sentDarkBackgroundColours()).toEqual([true, false]);
    });

    it('waits for the next compilation to send the background to a compiler that ignores it', () => {
        const {pane, sentDarkBackgroundColours} = makePane(false);

        pane.onThemeChange(themes.dark);
        expect(sentDarkBackgroundColours()).toEqual([]);

        pane.compile();
        expect(sentDarkBackgroundColours()).toEqual([true]);
    });
});
