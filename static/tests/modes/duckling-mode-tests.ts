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

import {ducklingCompletionItemProvider, ducklingIndentationRules} from '../../modes/duckling-mode.js';

describe('Duckling mode completions', () => {
    it('suggests keywords, types, and useful source snippets with the current word range', () => {
        const model = {
            getWordUntilPosition: () => ({startColumn: 2, endColumn: 5}),
            getValue: () => 'fun helper() -> i64 = { return 0; }',
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
        expect(byLabel.get('helper')).toMatchObject({
            kind: monaco.languages.CompletionItemKind.Function,
            detail: 'Function declared in this editor',
            insertText: 'helper',
        });
        expect(byLabel.get('fun main')).toMatchObject({
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText: 'fun main() -> i64 = {\n\treturn 0;\n}',
        });
        expect(byLabel.get('fun function')?.insertText).toContain(`\${1:name}`);
        expect(byLabel.get('import module')?.insertText).toBe(`import \${1:module} as \${2:Alias};`);
    });
});

describe('Duckling mode indentation', () => {
    it('increases indentation after an opening block or grouped expression, but not comments', () => {
        expect(ducklingIndentationRules.increaseIndentPattern.test('fun main() -> i64 = {')).toBe(true);
        expect(ducklingIndentationRules.increaseIndentPattern.test('let values = [')).toBe(true);
        expect(ducklingIndentationRules.increaseIndentPattern.test('call(')).toBe(true);
        expect(ducklingIndentationRules.increaseIndentPattern.test('# { comment')).toBe(false);
    });

    it('decreases indentation when a closing bracket starts the line', () => {
        expect(ducklingIndentationRules.decreaseIndentPattern.test('}')).toBe(true);
        expect(ducklingIndentationRules.decreaseIndentPattern.test('    ]')).toBe(true);
        expect(ducklingIndentationRules.decreaseIndentPattern.test(')')).toBe(true);
        expect(ducklingIndentationRules.decreaseIndentPattern.test('return 0;')).toBe(false);
    });
});
