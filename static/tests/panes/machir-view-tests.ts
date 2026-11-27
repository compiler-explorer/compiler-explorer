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

import {ParsedAsmResultLine} from '../../../types/asmresult/asmresult.interfaces.js';
import {MachIr} from '../../panes/machir-view.js';

/**
 * A MIRL listing cut down to the four shapes that decide whether a line may reach the editor: the user's own source,
 * a second file of the project, std, and an instruction with no position at all.
 */
const listing: ParsedAsmResultLine[] = [
    {text: 'pub fun example.example.main(_0: i64) i64 {'},
    {text: '    bind v2 = _0;'},
    {
        text: '    val _1: i64 = call std.math.min(4, _0) at l24;',
        source: {file: null, line: 12, column: 5, mainsource: true},
    },
    {text: '_2(_5: i64):'},
    {
        text: '    val _6: i64 = add(_5, 1) at l60 in s0;',
        source: {file: 'src/helper.mach', line: 2, column: 9},
    },
    {
        text: '    val _7: i1 = icmp.slt(_0, 4) at l52 in s1;',
        source: {file: '/opt/compiler-explorer/mach-7.2.0/std/src/math.mach', line: 16, column: 5},
    },
    {
        text: '    ret(_4) at l46;',
        source: {file: null, line: 14, column: 5, mainsource: true},
    },
];

function pane(settings: Record<string, unknown> = {}) {
    const view = Object.create(MachIr.prototype) as MachIr;
    Object.assign(view, {
        listing,
        compilerInfo: {compilerId: 1, editorId: 2, treeId: 0},
        editorDecorations: {set: vi.fn()},
        editor: {deltaDecorations: vi.fn().mockReturnValue([]), revealLinesInCenter: vi.fn()},
        eventHub: {emit: vi.fn(), on: vi.fn()},
        decorations: {},
        previousDecorations: [],
        settings,
    });
    return view;
}

describe('Mach IR pane', () => {
    it('announces nothing while the base constructor registers its callbacks', () => {
        // a synchronous reply to an emit from here would land before the pane's own fields are initialised
        const view = pane();
        Object.assign(view.editor, {onMouseMove: vi.fn(), onDidChangeCursorSelection: vi.fn()});
        view.registerCallbacks();
        expect(view.eventHub.emit).not.toHaveBeenCalled();
    });

    it('colours only the lines whose position is in the editor', () => {
        const view = pane();
        // the editor's lines 12 and 14 are coloured; helper.mach line 2 and math.mach line 16 are not the editor's
        view.onColours(2, {11: 0, 13: 1, 1: 2, 15: 3}, 'rainbow');

        const decorations = (view.editorDecorations.set as any).mock.calls[0][0];
        expect(decorations.map((d: any) => d.range.startLineNumber)).toEqual([3, 7]);
        expect(decorations.map((d: any) => d.options.className)).toEqual([
            'line-linkage rainbow-0',
            'line-linkage rainbow-1',
        ]);
    });

    it('leaves the editor alone for a line whose position is in another file', () => {
        const view = pane();
        // helper.mach:2 would mark the user's line 2 if the file were ignored
        view.onColours(2, {1: 0}, 'rainbow');
        expect((view.editorDecorations.set as any).mock.calls[0][0]).toEqual([]);
    });

    it('ignores colours meant for another editor', () => {
        const view = pane();
        view.onColours(9, {11: 0}, 'rainbow');
        expect(view.editorDecorations.set).not.toHaveBeenCalled();
    });

    it('links the editor line the hovered instruction came from', () => {
        const view = pane({hoverShowSource: true, indefiniteLineHighlight: true});
        view.onMouseMove({target: {position: {lineNumber: 3}}} as any);
        expect(view.eventHub.emit).toHaveBeenCalledWith('editorLinkLine', 2, 12, 5, 5, false);
    });

    it('links nothing when the hovered instruction came from another file', () => {
        const view = pane({hoverShowSource: true, indefiniteLineHighlight: true});
        view.onMouseMove({target: {position: {lineNumber: 5}}} as any);
        expect(view.eventHub.emit).toHaveBeenCalledWith('editorLinkLine', 2, -1, -1, -1, false);
    });

    it('marks every listing line an editor line produced, and none it did not', () => {
        const view = pane({indefiniteLineHighlight: true});
        view.onPanesLinkLine(1, 12, 5, 5, false, 'editor');

        const marked = (view.editor.deltaDecorations as any).mock.calls[0][1];
        expect(marked.map((d: any) => d.range.startLineNumber)).toEqual([3, 3]);

        // helper.mach's line 2, which is not the editor's line 2
        view.onPanesLinkLine(1, 2, 9, 9, false, 'editor');
        expect((view.editor.deltaDecorations as any).mock.calls[1][1]).toEqual([]);
    });
});
