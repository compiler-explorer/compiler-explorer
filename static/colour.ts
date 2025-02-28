// Copyright (c) 2016, Compiler Explorer Authors
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
import {Themes} from './themes.js';

export type ColourScheme =
    | 'rainbow'
    | 'rainbow2'
    | 'earth'
    | 'green-blue'
    | 'gray-shade'
    | 'rainbow-dark'
    | 'soft-rainbow-dark'
    | 'pink';

export type AppTheme = Themes | 'all';

export interface ColourSchemeInfo {
    name: ColourScheme;
    desc: string;
    count: number;
    themes: AppTheme[];
}

// If you want to use a scheme in every theme, set `theme: ['all']`
export const schemes: ColourSchemeInfo[] = [
    {name: 'rainbow', desc: 'Rainbow 1', count: 12, themes: ['default']},
    {name: 'rainbow2', desc: 'Rainbow 2', count: 12, themes: ['default']},
    {
        name: 'earth',
        desc: 'Earth tones (colourblind safe)',
        count: 9,
        themes: ['default'],
    },
    {
        name: 'green-blue',
        desc: 'Greens and blues (colourblind safe)',
        count: 4,
        themes: ['default'],
    },
    {
        name: 'gray-shade',
        desc: 'Gray shades',
        count: 4,
        themes: ['dark', 'darkplus', 'real-dark', 'onedark'],
    },
    {
        name: 'rainbow-dark',
        desc: 'Dark Rainbow',
        count: 12,
        themes: ['dark', 'darkplus', 'real-dark', 'onedark'],
    },
    {
        name: 'soft-rainbow-dark',
        desc: 'Soft Dark Rainbow',
        count: 11,
        themes: ['dark', 'darkplus', 'real-dark', 'onedark'],
    },
    {
        name: 'pink',
        desc: 'Pink',
        count: 5,
        themes: ['pink'],
    },
];

export function applyColours(
    colours: Record<number, number>,
    schemeName: string,
    editorDecorations: monaco.editor.IEditorDecorationsCollection,
): void {
    const scheme = schemes.find(scheme => scheme.name === schemeName) ?? schemes[0];
    const newDecorations: monaco.editor.IModelDeltaDecoration[] = Object.entries(colours).map(([line, index]) => {
        const realLineNumber = Number.parseInt(line) + 1;
        return {
            range: new monaco.Range(realLineNumber, 1, realLineNumber, 1),
            options: {
                isWholeLine: true,
                className: 'line-linkage ' + scheme.name + '-' + (index % scheme.count),
            },
        };
    });

    editorDecorations.set(newDecorations);
}
