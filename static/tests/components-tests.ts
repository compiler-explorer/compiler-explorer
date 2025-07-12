// Copyright (c) 2025, Compiler Explorer Authors
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

import {createComponentConfig, createLayoutItem, toGoldenLayoutConfig} from '../../static/components.js';

import {
    COMPILER_COMPONENT_NAME,
    CURRENT_LAYOUT_VERSION,
    EDITOR_COMPONENT_NAME,
} from '../../static/components.interfaces.js';

// Helper function to create versioned configs for testing
function createVersionedConfig(content: any[], additionalProps: any = {}) {
    return {
        version: CURRENT_LAYOUT_VERSION,
        content,
        ...additionalProps,
    };
}

describe('Component helper functions', () => {
    describe('createComponentConfig', () => {
        it('should create valid component config', () => {
            const config = createComponentConfig(EDITOR_COMPONENT_NAME, {id: 1, lang: 'c++'});
            expect(config).toEqual({
                type: 'component',
                componentName: EDITOR_COMPONENT_NAME,
                componentState: {id: 1, lang: 'c++'},
            });
        });
    });

    describe('createLayoutItem', () => {
        it('should create valid layout item', () => {
            const content = [
                createComponentConfig(EDITOR_COMPONENT_NAME, {}),
                createComponentConfig(COMPILER_COMPONENT_NAME, {source: 1, lang: 'c++'}),
            ];
            const layoutItem = createLayoutItem('row', content);
            expect(layoutItem).toEqual({
                type: 'row',
                content,
            });
        });
    });

    describe('toGoldenLayoutConfig', () => {
        it('should pass through config unchanged', () => {
            const config = createVersionedConfig(
                [
                    createLayoutItem('row', [
                        createComponentConfig(EDITOR_COMPONENT_NAME, {id: 1}),
                        createComponentConfig(COMPILER_COMPONENT_NAME, {source: 1, lang: 'c++'}),
                    ]),
                ],
                {
                    settings: {showPopoutIcon: false},
                },
            );
            const result = toGoldenLayoutConfig(config);
            expect(result).toBe(config);
        });
    });
});
