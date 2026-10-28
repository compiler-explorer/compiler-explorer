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

import {
    ducklingCompletionItemProvider,
    ducklingDefinitionProvider,
    ducklingIndentationRules,
} from '../../modes/duckling-mode.js';

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

describe('Duckling go to definition', () => {
    function createModel(source: string, word: string | null) {
        return {
            getValue: () => source,
            getWordAtPosition: () => (word ? {word, startColumn: 1, endColumn: word.length + 1} : null),
            uri: monaco.Uri.parse('inmemory://model/duckling'),
        } as unknown as monaco.editor.ITextModel;
    }

    it('navigates a function reference to its definition', () => {
        const model = createModel('fun main() = { return helper(); }\n    fun helper() = {}', 'helper');
        const definitions = ducklingDefinitionProvider.provideDefinition(model, new monaco.Position(1, 25));

        expect(definitions).toHaveLength(1);
        expect(definitions?.[0]).toMatchObject({
            uri: model.uri,
            range: {startLineNumber: 2, startColumn: 9, endLineNumber: 2, endColumn: 15},
        });
    });

    it('returns every matching definition when a name is ambiguous', () => {
        const model = createModel('fun helper() = {}\nfun helper() = {}', 'helper');
        const definitions = ducklingDefinitionProvider.provideDefinition(model, new monaco.Position(1, 5));

        expect(definitions).toHaveLength(2);
    });

    it('returns no location for an unknown name or a position without a word', () => {
        const source = 'fun main() = {}';
        expect(
            ducklingDefinitionProvider.provideDefinition(createModel(source, 'missing'), new monaco.Position(1, 1)),
        ).toEqual([]);
        expect(
            ducklingDefinitionProvider.provideDefinition(createModel(source, null), new monaco.Position(1, 1)),
        ).toBeNull();
    });
});
