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

import * as monaco from 'monaco-editor';
import {createInstantHoverDelegate} from 'monaco-editor/base/browser/ui/hover/hoverDelegateFactory.js';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {disposeDiffEditor} from '../monaco-config.js';

// A context menu builds its ActionBar through createInstantHoverDelegate().
function openHover() {
    createInstantHoverDelegate().dispose();
}

describe('disposeDiffEditor', () => {
    let editor: monaco.editor.IStandaloneCodeEditor;

    beforeEach(() => {
        // happy-dom has no 2D canvas, which Monaco reads the device pixel ratio from.
        vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as CanvasRenderingContext2D);
        editor = monaco.editor.create(document.createElement('div'));
    });

    afterEach(() => {
        editor.dispose();
        vi.restoreAllMocks();
    });

    it('leaves hovers broken when a diff editor is disposed directly', () => {
        monaco.editor.createDiffEditor(document.createElement('div')).dispose();
        expect(openHover).toThrow('InstantiationService has been disposed');
    });

    it('keeps hovers working in the remaining editors', () => {
        disposeDiffEditor(monaco.editor.createDiffEditor(document.createElement('div')));
        expect(openHover).not.toThrow();
    });
});
