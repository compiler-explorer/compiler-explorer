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

import type {CompilerInfo} from '../../../types/compiler.interfaces.js';
import type {Hub} from '../../hub.js';
import {CompilerPicker} from '../../widgets/compiler-picker.js';

const tomSelectOptions = vi.hoisted(() => vi.fn());
const popup = vi.hoisted(() => ({setLang: vi.fn(), show: vi.fn()}));

vi.mock('../../widgets/compiler-picker-popup.js', () => ({
    CompilerPickerPopup: class {
        setLang = popup.setLang;
        show = popup.show;
    },
}));

vi.mock('tom-select', () => ({
    default: class {
        dropdown_content = document.createElement('div');

        constructor(node: HTMLSelectElement, options: unknown) {
            tomSelectOptions(node, options);
        }

        on() {}
        close() {}
        destroy() {}
    },
}));

const flush = () => new Promise(resolve => setTimeout(resolve, 0));

function makePicker(getCompilersForLang: () => Promise<Record<string, Partial<CompilerInfo>>>) {
    const domRoot = $(
        '<div><select class="compiler-picker"></select><button class="picker-popout-button"></button></div>',
    );
    const hub = {
        createEventHub: () => ({on: vi.fn(), unsubscribe: vi.fn(), emit: vi.fn()}),
        compilerService: {getCompilersForLang, getGroupsInUse: vi.fn().mockResolvedValue([])},
    };
    const picker = new CompilerPicker(domRoot, hub as unknown as Hub, 'c++', 'gcc', vi.fn());
    return {domRoot, picker};
}

describe('CompilerPicker', () => {
    beforeEach(() => {
        tomSelectOptions.mockClear();
        popup.setLang.mockClear();
        popup.show.mockClear();
    });

    it('shows every compiler when more than 1000 are available', async () => {
        const picker = Object.create(CompilerPicker.prototype) as CompilerPicker;
        const options = Array.from({length: 1001}, (_, id) => ({
            id: String(id),
            name: String(id),
            $groups: [],
        })) as unknown as (CompilerInfo & {$groups: string[]})[];
        Object.assign(picker, {
            initNonce: 0,
            domNode: document.createElement('select'),
            compilerService: {getCompilersForLang: vi.fn().mockResolvedValue({})},
            getGroups: vi.fn().mockResolvedValue([]),
            getOptions: vi.fn().mockResolvedValue(options),
            popup: {setLang: vi.fn()},
            toolbarPopoutButton: $('<div>'),
        });

        await (picker as unknown as {initialize: (langId: string, compilerId: string) => Promise<void>}).initialize(
            'c++',
            '',
        );

        expect(tomSelectOptions).toHaveBeenCalledWith(
            picker.domNode,
            expect.objectContaining({maxOptions: options.length}),
        );
    });

    describe('after the compilers failed to load', () => {
        const compilers = {gcc: {id: 'gcc', name: 'GCC', group: 'gcc'}};

        it('retries and opens the popup when the popout button is clicked', async () => {
            const getCompilersForLang = vi
                .fn()
                .mockRejectedValueOnce(new Error('Failed to fetch'))
                .mockResolvedValue(compilers);
            const {domRoot, picker} = makePicker(getCompilersForLang);
            await flush();
            expect(picker.tomSelect).toBeNull();

            domRoot.find('.picker-popout-button').trigger('click');

            await vi.waitFor(() => expect(popup.show).toHaveBeenCalledTimes(1));
            expect(tomSelectOptions).toHaveBeenCalledTimes(1);
            expect(picker.tomSelect).not.toBeNull();
        });

        it('retries when the bare select is clicked', async () => {
            const getCompilersForLang = vi
                .fn()
                .mockRejectedValueOnce(new Error('Failed to fetch'))
                .mockResolvedValue(compilers);
            const {domRoot, picker} = makePicker(getCompilersForLang);
            await flush();

            domRoot.find('.compiler-picker').trigger('mousedown');

            await vi.waitFor(() => expect(picker.tomSelect).not.toBeNull());
            expect(popup.show).not.toHaveBeenCalled();
        });

        it('does nothing on a popout click while the compilers still fail to load', async () => {
            const getCompilersForLang = vi.fn().mockRejectedValue(new Error('Failed to fetch'));
            const {domRoot, picker} = makePicker(getCompilersForLang);
            await flush();

            domRoot.find('.picker-popout-button').trigger('click');
            await flush();

            expect(getCompilersForLang).toHaveBeenCalledTimes(2);
            expect(picker.tomSelect).toBeNull();
            expect(popup.show).not.toHaveBeenCalled();
        });

        it('neither rebuilds nor opens the popup if the picker is destroyed while retrying', async () => {
            let finishRetry: (value: typeof compilers) => void = () => {};
            const getCompilersForLang = vi
                .fn()
                .mockRejectedValueOnce(new Error('Failed to fetch'))
                .mockReturnValueOnce(
                    new Promise(resolve => {
                        finishRetry = resolve;
                    }),
                )
                .mockResolvedValue(compilers);
            const {domRoot, picker} = makePicker(getCompilersForLang);
            await flush();

            domRoot.find('.picker-popout-button').trigger('click');
            picker.destroy();
            finishRetry(compilers);
            await flush();

            expect(tomSelectOptions).not.toHaveBeenCalled();
            expect(popup.show).not.toHaveBeenCalled();
        });
    });
});
