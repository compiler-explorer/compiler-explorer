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

import $ from 'jquery';
import * as monaco from 'monaco-editor';
import {afterEach, describe, expect, it, vi} from 'vitest';

import type {CompilerState} from '../../panes/compiler.interfaces.js';
import {Compiler} from '../../panes/compiler.js';
import * as fetchJson from '../../services/fetch-json.js';
import {type LanguageMap, LanguagesService, languagesService} from '../../services/languages.service.js';

vi.mock('../../widgets/compiler-picker.js', () => ({
    CompilerPicker: class {
        update = vi.fn();
    },
}));

describe('Compiler output editor', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('does not enable color decorators for assembly output', () => {
        const createEditor = vi
            .spyOn(monaco.editor, 'create')
            .mockImplementation(() => ({}) as monaco.editor.IStandaloneCodeEditor);
        const compiler = Object.create(Compiler.prototype) as Compiler;

        compiler.createEditor(document.createElement('div'));

        expect(createEditor).toHaveBeenCalledWith(
            expect.anything(),
            expect.objectContaining({
                colorDecorators: false,
                language: 'asm',
                readOnly: true,
            }),
        );
    });
});

describe('Compiler default options', () => {
    afterEach(() => {
        vi.clearAllTimers();
        vi.useRealTimers();
        vi.restoreAllMocks();
        $(document).off('keyup.editable').off('click');
    });

    function compilerWithOptions(initialOptions: string) {
        vi.spyOn(languagesService, 'getLanguagesOrFail').mockReturnValue({
            'c++': {id: 'c++', defaultOptions: '-O2'},
        } as LanguageMap);
        const compiler = Object.create(Compiler.prototype) as Compiler;
        const optionsField = $('<input>').val(initialOptions);
        const sentOptions: string[] = [];
        Object.assign(compiler, {
            options: initialOptions,
            optionsChangedBeforeInitialStateLoad: false,
            initialStateLoaded: false,
            optionsField,
            filters: {on: vi.fn()},
            fullTimingInfo: $('<div>'),
            compileClearCache: $('<button>'),
            compilerLicenseButton: $('<button>'),
            libsButton: $('<button>'),
            editor: {onMouseMove: vi.fn(), onDidChangeCursorSelection: vi.fn(), onMouseUp: vi.fn()},
            hub: {deferred: false},
            eventHub: {emit: vi.fn()},
            sourceEditorId: 1,
            initLangAndCompiler: vi.fn(async () => {
                Reflect.set(compiler, 'currentLangId', 'c++');
            }),
            initLibraries: vi.fn(),
            sendCompiler: vi.fn(() => sentOptions.push(Reflect.get(compiler, 'options'))),
            updateCompilerInfo: vi.fn(),
            updateButtons: vi.fn(),
            initToolButtons: vi.fn(),
            updateState: vi.fn(),
            compile: vi.fn(),
            undefer: vi.fn(),
        });
        return {compiler, optionsField, displayedOptions: () => optionsField.val(), sentOptions};
    }

    it('requests default options with language metadata', async () => {
        const fetch = vi.spyOn(fetchJson, 'fetchJson').mockResolvedValue([{id: 'c++', defaultOptions: '-O2'}]);
        const service = new LanguagesService();

        expect((await service.getLanguages())['c++']?.defaultOptions).toBe('-O2');
        expect(fetch).toHaveBeenCalledWith(expect.stringContaining(',defaultOptions&hash='));
    });

    it.each([
        [undefined, '-O2'],
        ['', ''],
        ['-O3', '-O3'],
    ])('initializes stored options %s as %s before sending the compiler', async (storedOptions, expectedOptions) => {
        const fixture = compilerWithOptions(storedOptions ?? '');
        const state: CompilerState = {compiler: 'gdefault', options: storedOptions};

        await Reflect.get(fixture.compiler, 'postInit').call(fixture.compiler, state);

        expect(fixture.displayedOptions()).toBe(expectedOptions);
        expect(fixture.sentOptions).toEqual([expectedOptions]);
    });

    it.each([
        [['-Og'], '-Og'],
        [['-Og', ''], ''],
    ])('preserves raw options input before the debounce runs: %s', async (edits, expectedOptions) => {
        vi.useFakeTimers();
        const fixture = compilerWithOptions('');
        fixture.compiler.initCallbacks();
        const optionsChanged = vi.spyOn(fixture.compiler, 'onOptionsChange');
        let finishInit = () => {};
        Reflect.set(
            fixture.compiler,
            'initLangAndCompiler',
            () =>
                new Promise<void>(resolve => {
                    finishInit = () => {
                        Reflect.set(fixture.compiler, 'currentLangId', 'c++');
                        resolve();
                    };
                }),
        );
        const initializing = Reflect.get(fixture.compiler, 'postInit').call(fixture.compiler, {
            compiler: 'gdefault',
        });

        vi.advanceTimersByTime(100);
        for (const edit of edits) {
            fixture.optionsField.val(edit).trigger('input');
        }
        vi.advanceTimersByTime(100);
        expect(optionsChanged).not.toHaveBeenCalled();
        finishInit();
        await initializing;

        expect(fixture.displayedOptions()).toBe(expectedOptions);
        expect(optionsChanged).not.toHaveBeenCalled();
    });

    it.each(['-Og', ''])('persists and compiles options %s before initial state loads', async expectedOptions => {
        const fixture = compilerWithOptions('-O3');
        fixture.compiler.initCallbacks();
        let finishInit = () => {};
        Reflect.set(
            fixture.compiler,
            'initLangAndCompiler',
            () =>
                new Promise<void>(resolve => {
                    finishInit = () => {
                        Reflect.set(fixture.compiler, 'currentLangId', 'c++');
                        resolve();
                    };
                }),
        );
        const initializing = Reflect.get(fixture.compiler, 'postInit').call(fixture.compiler, {
            compiler: 'gdefault',
        });

        fixture.optionsField.val(expectedOptions).trigger('input');
        await vi.waitFor(
            () => {
                expect(fixture.compiler.updateState).toHaveBeenCalledTimes(1);
                expect(fixture.compiler.compile).toHaveBeenCalledTimes(1);
                expect(fixture.sentOptions).toEqual([expectedOptions]);
            },
            {timeout: 2000},
        );
        finishInit();
        await initializing;

        expect(fixture.displayedOptions()).toBe(expectedOptions);
        expect(fixture.sentOptions).toEqual([expectedOptions, expectedOptions]);
    });
});

describe('Duckling output pane events', () => {
    function compilerWithDucklingOutputs(supportedOutputKinds: string[]) {
        const compiler = Object.create(Compiler.prototype) as Compiler;
        const ducklingOutputButtons = new Map([
            ['hir', $('<button>')],
            ['mir', $('<button>')],
        ]);
        Object.assign(compiler, {
            id: 12,
            ducklingOutputViewsOpen: new Set(),
            ducklingOutputButtons,
            compiler: {supportsDucklingOutputs: supportedOutputKinds},
            compile: vi.fn(),
        });
        return {compiler, ducklingOutputButtons};
    }

    it('tracks and compiles each supported output pane independently', () => {
        const {compiler, ducklingOutputButtons} = compilerWithDucklingOutputs(['hir', 'mir']);

        compiler.onDucklingOutputViewOpened(12, 'hir');
        compiler.onDucklingOutputViewOpened(12, 'mir');

        expect(Reflect.get(compiler, 'ducklingOutputViewsOpen')).toEqual(new Set(['hir', 'mir']));
        expect(ducklingOutputButtons.get('hir')?.prop('disabled')).toBe(true);
        expect(ducklingOutputButtons.get('mir')?.prop('disabled')).toBe(true);
        expect(compiler.compile).toHaveBeenCalledTimes(2);
    });

    it('ignores unsupported outputs and re-enables only the closed pane button', () => {
        const {compiler, ducklingOutputButtons} = compilerWithDucklingOutputs(['hir', 'mir']);
        compiler.onDucklingOutputViewOpened(12, 'hir');
        compiler.onDucklingOutputViewOpened(12, 'parser');
        compiler.onDucklingOutputViewOpened(12, 'mir');

        compiler.onDucklingOutputViewClosed(12, 'hir');

        expect(Reflect.get(compiler, 'ducklingOutputViewsOpen')).toEqual(new Set(['mir']));
        expect(ducklingOutputButtons.get('hir')?.prop('disabled')).toBe(false);
        expect(ducklingOutputButtons.get('mir')?.prop('disabled')).toBe(true);
        expect(compiler.compile).toHaveBeenCalledTimes(3);
    });
});
