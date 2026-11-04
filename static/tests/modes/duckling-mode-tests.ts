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
import {getEnterAction} from 'monaco-editor/esm/vs/editor/common/languages/enterAction.js';
import {ILanguageConfigurationService} from 'monaco-editor/esm/vs/editor/common/languages/languageConfigurationRegistry.js';
import {getReindentEditOperations} from 'monaco-editor/esm/vs/editor/contrib/indentation/common/indentation.js';
import {StandaloneServices} from 'monaco-editor/esm/vs/editor/standalone/browser/standaloneServices.js';
import {describe, expect, it} from 'vitest';

import '../../modes/duckling-mode.js';

function tokens(source: string): [string, string][][] {
    const lines = source.split('\n');
    return monaco.editor
        .tokenize(source, 'duckling')
        .map((row, index) =>
            row
                .map((token, i): [string, string] => [
                    lines[index].slice(token.offset, row[i + 1]?.offset),
                    token.type.replace(/\.duckling$/, ''),
                ])
                .filter(([text]) => text.trim() !== ''),
        );
}

describe('Duckling names', () => {
    it('distinguishes word operators and predefined names', () => {
        expect(tokens('return ptrof self as type; true false none')[0]).toEqual([
            ['return', 'keyword'],
            ['ptrof', 'keyword.operator'],
            ['self', 'variable.predefined'],
            ['as', 'keyword.operator'],
            ['type', 'type.identifier'],
            [';', 'delimiter'],
            ['true', 'constant.language'],
            ['false', 'constant.language'],
            ['none', 'constant.predefined'],
        ]);
    });

    it('highlights primitive and conventional library types', () => {
        const types = ['i64', 'str', 'type', 'void', 'Array', 'Dict', 'Set', 'f80'];
        expect(tokens(types.join(' '))[0]).toEqual(types.map(type => [type, 'type.identifier']));
    });

    it('recognises declarations, attributes and calls without overriding keywords and types', () => {
        expect(tokens('public fun décrire() -> i64; class Sample; namespace Demo;')[0]).toEqual([
            ['public', 'keyword'],
            ['fun', 'keyword'],
            ['décrire', 'entity.name.function'],
            ['()', 'delimiter.parenthesis'],
            ['->', 'operator'],
            ['i64', 'type.identifier'],
            [';', 'delimiter'],
            ['class', 'keyword'],
            ['Sample', 'type.identifier'],
            [';', 'delimiter'],
            ['namespace', 'keyword'],
            ['Demo', 'type.identifier'],
            [';', 'delimiter'],
        ]);
        expect(tokens('@inline décrire(); i64(); if();')[0]).toEqual([
            ['@inline', 'annotation'],
            ['décrire', 'entity.name.function'],
            ['()', 'delimiter.parenthesis'],
            [';', 'delimiter'],
            ['i64', 'type.identifier'],
            ['()', 'delimiter.parenthesis'],
            [';', 'delimiter'],
            ['if', 'keyword'],
            ['()', 'delimiter.parenthesis'],
            [';', 'delimiter'],
        ]);
        expect(tokens('fundecl prefix `; fun suffix .`;')[0]).toEqual([
            ['fundecl', 'keyword'],
            ['prefix', 'keyword'],
            ['`', 'operator'],
            [';', 'delimiter'],
            ['fun', 'keyword'],
            ['suffix', 'keyword'],
            ['.`', 'operator'],
            [';', 'delimiter'],
        ]);
    });
});

describe('Duckling numeric literals', () => {
    it.each([
        ['42', 'i32', 'number'],
        ['0xFF', 'u64', 'number.hex'],
        ['0B1010', 'u8', 'number.binary'],
        ['0o755', 'u16', 'number.octal'],
        ['.5e+2', 'f32', 'number'],
        ['1.e-3', 'f64', 'number'],
        ['1', 'f16', 'number'],
        ['1', 'f128', 'number'],
    ])('highlights %s with its %s suffix', (literal, suffix, type) => {
        expect(tokens(literal + suffix)[0]).toEqual([
            [literal, type],
            [suffix, 'type.identifier'],
        ]);
    });

    it('recovers from malformed suffixes without classifying them as numeric types', () => {
        expect(tokens('1. 1f80 0xFF_00 0b10_01 1u64tail')[0]).toEqual([
            ['1.', 'number'],
            ['1f80', 'number'],
            ['0xFF_00', 'number.hex'],
            ['0b10_01', 'number.binary'],
            ['1u64tail', 'number'],
        ]);
    });
});

describe('Duckling operators', () => {
    it('highlights builtin and custom symbol sequences', () => {
        const operators = ['+=', '**', '<<=', '::', '.', '.?', '.*', '+*', '\\', '`', ':`', '->`'];
        expect(tokens(operators.join(' '))[0]).toEqual(operators.map(operator => [operator, 'operator']));
    });

    it.each(['..', '.+', '.`'])('keeps %s separate from adjacent numbers', operator => {
        expect(tokens(`1${operator}2`)[0]).toEqual([
            ['1', 'number'],
            [operator, 'operator'],
            ['2', 'number'],
        ]);
    });
});

describe('Duckling comments', () => {
    it('keeps block comments across lines and resumes after the closing delimiter', () => {
        expect(tokens('#{ #{ comment\nreturn 1;\n#} let')).toEqual([
            [['#{ #{ comment', 'comment']],
            [['return 1;', 'comment']],
            [
                ['#}', 'comment'],
                ['let', 'keyword'],
            ],
        ]);
    });

    it('closes an empty block comment before following code', () => {
        expect(tokens('#{#} let')[0]).toEqual([
            ['#{#}', 'comment'],
            ['let', 'keyword'],
        ]);
    });
});

describe('Duckling strings', () => {
    it.each([
        ['"', 'string'],
        ["'", 'string.char'],
        ['f"', 'string'],
    ])('handles escaped and unescaped line endings in %s literals', (opening, type) => {
        expect(tokens(`${opening}unfinished\nlet`)[1]).toEqual([['let', 'keyword']]);
        expect(tokens(`${opening}continued\\\ntext${opening.slice(-1)}; let`)[1]).toEqual([
            [`text${opening.slice(-1)}`, type],
            [';', 'delimiter'],
            ['let', 'keyword'],
        ]);
        expect(tokens(`${opening}escaped\\\\\nlet`)[1]).toEqual([['let', 'keyword']]);
    });

    it('keeps code-like text inside strings and comments', () => {
        expect(tokens('"fun hidden() # \\" quote"; visible() # hidden()')[0]).toEqual([
            ['"fun hidden() # ', 'string'],
            ['\\"', 'string.escape'],
            [' quote"', 'string'],
            [';', 'delimiter'],
            ['visible', 'entity.name.function'],
            ['()', 'delimiter.parenthesis'],
            ['# hidden()', 'comment'],
        ]);
    });

    it('balances interpolation around nested braces, formatted strings, quoted text and comments', () => {
        const [expression, tail] = tokens('f"{f"{ {value} }" + "}" + #{ } #} # }\n + call()} tail"; after()');
        expect(expression).toContainEqual(['value', 'identifier']);
        expect(expression).toContainEqual(['"}"', 'string']);
        expect(expression).toContainEqual(['#{ } #}', 'comment']);
        expect(expression).toContainEqual(['# }', 'comment']);
        expect(tail).toEqual([
            ['+', 'operator'],
            ['call', 'entity.name.function'],
            ['()', 'delimiter.parenthesis'],
            ['}', 'delimiter.curly'],
            [' tail"', 'string'],
            [';', 'delimiter'],
            ['after', 'entity.name.function'],
            ['()', 'delimiter.parenthesis'],
        ]);
    });
});

describe('Duckling indentation', () => {
    it('indents blocks and grouped expressions without treating strings and comments as brackets', () => {
        const source = [
            'fun main() -> i64 = { # block',
            '    # a comment ending with {',
            '    let text = "{";',
            '    let values = [',
            '        call(',
            '            1,',
            '        ),',
            '    ];',
            '    return 0;',
            '}',
        ].join('\n');
        const model = monaco.editor.createModel(source.replace(/^ +/gm, ''), 'duckling');
        model.updateOptions({tabSize: 4, indentSize: 4, insertSpaces: true});
        const configuration = StandaloneServices.get(ILanguageConfigurationService);
        try {
            const column = model.getLineMaxColumn(1);
            expect(
                getEnterAction(
                    monaco.editor.EditorAutoIndentStrategy.Advanced,
                    model,
                    new monaco.Range(1, column, 1, column),
                    configuration,
                )?.indentAction,
            ).toBe(monaco.languages.IndentAction.Indent);
            model.applyEdits(getReindentEditOperations(model, configuration, 1, model.getLineCount()));
            expect(model.getValue()).toBe(source);
        } finally {
            model.dispose();
        }
    });
});
