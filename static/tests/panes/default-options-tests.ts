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
import {afterEach, describe, expect, it, vi} from 'vitest';

import {Conformance} from '../../panes/conformance-view.js';
import type {ExecutorState} from '../../panes/executor.interfaces.js';
import {Executor} from '../../panes/executor.js';
import {type LanguageMap, languagesService} from '../../services/languages.service.js';

vi.mock('../../components.js', async importOriginal => ({
    ...(await importOriginal<typeof import('../../components.js')>()),
    createDragSource: vi.fn(),
}));

vi.mock('../../widgets/compiler-picker.js', () => ({
    CompilerPicker: class {
        update = vi.fn();
    },
}));

vi.mock('../../widgets/compilation-options.js', () => ({
    BaseCompilationOptions: class {
        clear = vi.fn();
    },
}));

function mockLanguages() {
    vi.spyOn(languagesService, 'getLanguagesOrFail').mockReturnValue({
        'c++': {id: 'c++', defaultOptions: '-O2'},
    } as LanguageMap);
}

describe('Executor default options', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    function executorWithOptions(initialOptions: string) {
        mockLanguages();
        const executor = Object.create(Executor.prototype) as Executor;
        const optionsField = $('<input>').val(initialOptions);
        Object.assign(executor, {
            options: initialOptions,
            optionsChangedBeforeInitialStateLoad: false,
            initialStateLoaded: false,
            optionsField,
            domRoot: $('<div>'),
            hub: {deferred: true},
            eventHub: {emit: vi.fn()},
            sourceTreeId: null,
            initLangAndCompiler: vi.fn(async () => {
                Reflect.set(executor, 'currentLangId', 'c++');
            }),
            initLibraries: vi.fn(),
            updateCompilerInfo: vi.fn(),
            updateState: vi.fn(),
            doTypicalOnChange: vi.fn(),
            compile: vi.fn(),
        });
        return {executor, displayedOptions: () => optionsField.val()};
    }

    it.each([
        [undefined, '-O2'],
        ['', ''],
        ['-O3', '-O3'],
    ])('initializes stored options %s as %s', async (storedOptions, expectedOptions) => {
        const fixture = executorWithOptions(storedOptions ?? '');
        const state = {compiler: 'gdefault', options: storedOptions} as ExecutorState;

        await Reflect.get(fixture.executor, 'postInit').call(fixture.executor, state);

        expect(Reflect.get(fixture.executor, 'options')).toBe(expectedOptions);
        expect(fixture.displayedOptions()).toBe(expectedOptions);
    });

    it('keeps options edited before the initial state loads', async () => {
        const fixture = executorWithOptions('');
        let finishInit = () => {};
        Reflect.set(
            fixture.executor,
            'initLangAndCompiler',
            () =>
                new Promise<void>(resolve => {
                    finishInit = () => {
                        Reflect.set(fixture.executor, 'currentLangId', 'c++');
                        resolve();
                    };
                }),
        );
        const initializing = Reflect.get(fixture.executor, 'postInit').call(fixture.executor, {compiler: 'gdefault'});

        fixture.executor.onOptionsChange('-Og');
        finishInit();
        await initializing;

        expect(Reflect.get(fixture.executor, 'options')).toBe('-Og');
    });
});

describe('Conformance view default options', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    function conformanceView() {
        mockLanguages();
        const view = Object.create(Conformance.prototype) as Conformance;
        Object.assign(view, {
            langId: 'c++',
            hub: {},
            container: {layoutManager: {}},
            compilerInfo: {editorId: 1},
            compilerPickers: [],
            selectorTemplate: $(
                '<div><input class="conformance-options"><button class="close-compiler"></button>' +
                    '<button class="copy-compiler"></button><button class="extract-compiler"></button></div>',
            ),
            selectorList: $('<div>'),
            saveState: vi.fn(),
            compileChild: vi.fn(),
            updateLibraries: vi.fn(),
            handleToolbarUI: vi.fn(),
        });
        return view;
    }

    const entryOptions = (view: Conformance) =>
        Reflect.get(view, 'compilerPickers').map((entry: {optionsField: JQuery}) => entry.optionsField.val());

    it("starts a new compiler entry with the language's default options", () => {
        const view = conformanceView();

        view.addCompilerPicker();

        expect(entryOptions(view)).toEqual(['-O2']);
    });

    it.each(['', '-O3'])('keeps the options %s of a restored or copied entry', options => {
        const view = conformanceView();

        view.addCompilerPicker({compilerId: 'gdefault', options});

        expect(entryOptions(view)).toEqual([options]);
    });
});
