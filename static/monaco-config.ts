// Copyright (c) 2021, Compiler Explorer Authors
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

import _ from 'underscore';
import * as monaco from 'monaco-editor';

import {SiteSettings} from './settings';

const DEFAULT_MONACO_CONFIG = {
    fontFamily: 'Consolas, "Liberation Mono", Courier, monospace',
    scrollBeyondLastLine: true,
    quickSuggestions: false,
    fixedOverflowWidgets: true,
    minimap: {
        maxColumn: 80,
    },
    folding: true,
    lineNumbersMinChars: 1,
    emptySelectionClipboard: true,
};

type EditorKinds = monaco.editor.IStandaloneDiffEditor | monaco.editor.IStandaloneCodeEditor;

/** Pick construction options based on editor kind */
type EditorConstructionType<E extends EditorKinds> = E extends monaco.editor.IStandaloneDiffEditor
    ? monaco.editor.IDiffEditorConstructionOptions
    : monaco.editor.IStandaloneEditorConstructionOptions;

/**
 * Extend the default monaco editor construction options.
 *
 * Type parameter E indicates which editor kind you're constructing a config
 * for. Valid options are EditorKinds, aka monaco.editor.IStandaloneDiffEditor
 * or monaco.editor.IStandaloneCodeEditor
 *
 * TODO(supergrecko): underscore.extend yields any, can we improve the type
 *  check on this?
 */
export function extendConfig<
    E extends EditorKinds = monaco.editor.IStandaloneCodeEditor,
    T = EditorConstructionType<E>
>(overrides: T, settings?: Pick<SiteSettings, 'editorsFFont' | 'autoIndent' | 'useVim' | 'editorsFLigatures'>): T {
    if (settings !== undefined) {
        return _.extend(
            {},
            DEFAULT_MONACO_CONFIG,
            {
                fontFamily: settings.editorsFFont,
                autoIndent: settings.autoIndent ? 'advanced' : 'none',
                vimInUse: settings.useVim,
                fontLigatures: settings.editorsFLigatures,
            },
            overrides
        );
    }
    return _.extend({}, DEFAULT_MONACO_CONFIG, overrides);
}
