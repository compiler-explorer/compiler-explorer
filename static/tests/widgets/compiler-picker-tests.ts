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
import {CompilerPicker} from '../../widgets/compiler-picker.js';

const tomSelectOptions = vi.hoisted(() => vi.fn());

vi.mock('tom-select', () => ({
    default: class {
        dropdown_content = document.createElement('div');

        constructor(node: HTMLSelectElement, options: unknown) {
            tomSelectOptions(node, options);
        }

        on() {}
    },
}));

describe('CompilerPicker', () => {
    beforeEach(() => {
        tomSelectOptions.mockClear();
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
});
