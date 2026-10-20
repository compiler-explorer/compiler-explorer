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

import {describe, expect, it, vi} from 'vitest';

// the editor module builds one of these at import time, and its constructor asks the server for the builtin sources
vi.mock('../../widgets/load-save.js', () => ({LoadSave: class {}}));
vi.mock('monaco-vim', () => ({initVimMode: () => undefined}));

import {ResultLine} from '../../../types/resultline/resultline.interfaces.js';
import {Editor} from '../../panes/editor.js';

/**
 * What the parser hands the editor for rustc 1.96 building `example.rs` with `mod a;`: an error in the main source, a
 * note in the crate's `a/mod.rs`, and a note inside the standard library, which lies outside the directory the crate
 * was written to (see test/rust/diagnostics).
 */
const diagnostics: (ResultLine & {sourcePane: string})[] = [
    {
        text: 'error[E0308]: mismatched types',
        sourcePane: 'rustc #1',
        tag: {file: 'example.rs', line: 9, column: 10, text: 'error[E0308]: mismatched types', severity: 3},
    },
    {
        text: 'note: function defined here',
        sourcePane: 'rustc #1',
        tag: {file: 'a/mod.rs', line: 1, column: 8, text: 'note: function defined here', severity: 1},
    },
    {
        text: 'note: method defined here',
        sourcePane: 'rustc #1',
        tag: {
            file: '../opt/compiler-explorer/rust-1.96.0/lib/rustlib/src/rust/library/alloc/src/vec/mod.rs',
            line: 1003,
            column: 12,
            text: 'note: method defined here',
            severity: 1,
        },
    },
];

/** The tree of the crate: `example.rs` in one editor, `a/mod.rs` in another, and `b/mod.rs` in a third. */
const tree = {
    multifileService: {
        getEditorIdByFilename: (filename: string) =>
            ({'example.rs': 1, 'a/mod.rs': 2, 'b/mod.rs': 3})[filename] ?? null,
        getMainSourceEditorId: () => 1,
    },
};

function pane(id: number, trees: any[] = []) {
    const view = Object.create(Editor.prototype) as Editor;
    Object.assign(view, {
        id,
        hub: {trees},
        editor: {getModel: () => null},
        getTokenSpan: () => ({colBegin: 0, colEnd: 0}),
        currentLanguage: {extensions: ['.rs']},
    });
    return view;
}

function marked(view: Editor, output = diagnostics) {
    return view.collectOutputWidgets(output).widgets.map(w => [w.startLineNumber, w.message]);
}

describe('Editor diagnostics', () => {
    it('marks nothing from outside the compiled sources when there is no tree', () => {
        // a single-file compile: the standard library's note would otherwise mark line 1003 of the user's file
        const single = diagnostics.filter(d => d.tag?.file !== 'a/mod.rs');
        expect(marked(pane(1), single)).toEqual([[9, 'error[E0308]: mismatched types']]);
    });

    it('marks the editor holding the file a diagnostic names, subdirectories included', () => {
        expect(marked(pane(1, [tree]))).toEqual([[9, 'error[E0308]: mismatched types']]);
        expect(marked(pane(2, [tree]))).toEqual([[1, 'note: function defined here']]);
    });

    it('marks no editor of a tree for a file the crate does not hold', () => {
        // b/mod.rs shares a basename with a/mod.rs and with the standard library's vec/mod.rs
        expect(marked(pane(3, [tree]))).toEqual([]);
    });

    it('still marks a single editor for a file named other than the compiled one', () => {
        // co2 compiles `example.rs` but tags its diagnostics with the user's `example.co2`
        const co2: (ResultLine & {sourcePane: string})[] = [
            {
                text: 'error: expected `;`',
                sourcePane: 'co2 #1',
                tag: {file: 'example.co2', line: 3, column: 1, text: 'error: expected `;`', severity: 3},
            },
        ];
        expect(marked(pane(1), co2)).toEqual([[3, 'error: expected `;`']]);
    });
});
