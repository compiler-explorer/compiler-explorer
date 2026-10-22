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
import {describe, expect, it} from 'vitest';

import {ducklingCompletionItemProvider} from '../../modes/duckling-mode.js';

describe('Duckling mode completions', () => {
    it('suggests keywords, types, and useful source snippets with the current word range', () => {
        const model = {
            getWordUntilPosition: () => ({startColumn: 2, endColumn: 5}),
        } as unknown as monaco.editor.ITextModel;
        const position = {lineNumber: 3, column: 5} as monaco.Position;

        const {suggestions} = ducklingCompletionItemProvider.provideCompletionItems(model, position);
        const byLabel = new Map(suggestions.map(suggestion => [suggestion.label, suggestion]));

        expect(byLabel.get('return')).toMatchObject({
            kind: monaco.languages.CompletionItemKind.Keyword,
            insertText: 'return',
            range: {startLineNumber: 3, endLineNumber: 3, startColumn: 2, endColumn: 5},
        });
        expect(byLabel.get('i64')).toMatchObject({
            kind: monaco.languages.CompletionItemKind.Class,
            insertText: 'i64',
        });
        expect(byLabel.get('fun main')).toMatchObject({
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText: 'fun main() -> i64 = {\n\treturn 0;\n}',
        });
        expect(byLabel.get('fun function')?.insertText).toContain(`\${1:name}`);
        expect(byLabel.get('import module')?.insertText).toBe(`import \${1:module} as \${2:Alias};`);
    });
});
