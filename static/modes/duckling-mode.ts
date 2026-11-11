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
import {ILanguageConfigurationService} from 'monaco-editor/esm/vs/editor/common/languages/languageConfigurationRegistry.js';
import {getReindentEditOperations} from 'monaco-editor/esm/vs/editor/contrib/indentation/common/indentation.js';
import {StandaloneServices} from 'monaco-editor/esm/vs/editor/standalone/browser/standaloneServices.js';

monaco.languages.register({id: 'duckling'});

const identifierCases = {
    '@typeKeywords': 'type.identifier',
    '@wordOperators': 'keyword.operator',
    '@keywords': 'keyword',
    'true|false': 'constant.language',
    self: 'variable.predefined',
    none: 'constant.predefined',
};

monaco.languages.setMonarchTokensProvider('duckling', {
    unicode: true,
    includeLF: true,
    keywords: [
        'assert',
        'block',
        'break',
        'case',
        'catch',
        'class',
        'compile_assert',
        'const',
        'continue',
        'debug',
        'defer',
        'destroy',
        'else',
        'expand',
        'export',
        'extends',
        'extern',
        'for',
        'fun',
        'fundecl',
        'global',
        'hides',
        'if',
        'implements',
        'import',
        'in',
        'lambda',
        'let',
        'loop',
        'match',
        'namespace',
        'pattern',
        'private',
        'protected',
        'public',
        'redo',
        'restart',
        'return',
        'switch',
        'template',
        'then',
        'throw',
        'try',
        'using',
        'var',
        'while',
        'with',
    ],
    wordOperators: [
        'and',
        'as',
        'box',
        'copy',
        'copyof',
        'cptr',
        'manyptr',
        'move',
        'new',
        'not',
        'or',
        'ptr',
        'ptrof',
        'ref',
        'refof',
        'sizeof',
        'slice',
        'xor',
    ],
    typeKeywords: [
        'i8',
        'i16',
        'i32',
        'i64',
        'i128',
        'u8',
        'u16',
        'u32',
        'u64',
        'u128',
        'f16',
        'f32',
        'f64',
        'f80',
        'f128',
        'char',
        'bool',
        'str',
        'type',
        'void',
        'Array',
        'Dict',
        'Set',
    ],
    identifier: String.raw`[_\p{L}][\w\p{L}\p{N}\p{M}]*`,
    operatorChar: /[=><!~?:&|+\-*/^%.\\`]/,
    numericSuffix: String.raw`(?:[iu](?:8|16|32|64|128)|f(?:16|32|64|128))(?![\w\p{L}\p{N}\p{M}])`,
    numericRecovery: String.raw`[_\p{L}\p{N}\p{M}]*`,
    tokenizer: {
        root: [
            [/#\{/, 'comment', '@blockComment'],
            [/#[^\n]*/, 'comment'],
            {include: '@whitespace'},
            [/f"/, 'string', '@formattedString'],
            [/"/, 'string', '@doubleString'],
            [/'/, 'string.char', '@singleString'],
            [/@\s*@identifier/, 'annotation'],
            [/(fun|fundecl)(\s+)(prefix|suffix)\b(?=\s*@operatorChar)/, ['keyword', 'white', 'keyword']],
            [/(fun|fundecl)(\s+)(@identifier)/, ['keyword', 'white', 'entity.name.function']],
            [/(class|namespace)(\s+)(@identifier)/, ['keyword', 'white', 'type.identifier']],
            [/@identifier(?=\s*\()/, {cases: {...identifierCases, '@default': 'entity.name.function'}}],
            [/@identifier/, {cases: {...identifierCases, '@default': 'identifier'}}],
            [/(0[xX][0-9a-fA-F]*)(@numericSuffix?)(@numericRecovery)/, ['number.hex', 'type.identifier', 'number.hex']],
            [/(0[bB][01]*)(@numericSuffix?)(@numericRecovery)/, ['number.binary', 'type.identifier', 'number.binary']],
            [/(0[oO][0-7]*)(@numericSuffix?)(@numericRecovery)/, ['number.octal', 'type.identifier', 'number.octal']],
            [
                /((?:\d+(?:\.(?!@operatorChar)\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(@numericSuffix?)(@numericRecovery)/,
                ['number', 'type.identifier', 'number'],
            ],
            [/@operatorChar+/, 'operator'],
            [/[{}()[\]]/, '@brackets'],
            [/[;,]/, 'delimiter'],
        ],
        whitespace: [[/[ \t\r\n]+/, 'white']],
        blockComment: [
            [/#\}/, 'comment', '@pop'],
            [/[^#]+/, 'comment'],
            [/#/, 'comment'],
        ],
        doubleString: [{include: '@stringCommon'}, [/[^\\"\n]+/, 'string'], [/"/, 'string', '@pop']],
        singleString: [{include: '@stringCommon'}, [/[^\\'\n]+/, 'string.char'], [/'/, 'string.char', '@pop']],
        stringCommon: [
            [/\\[\s\S]/, 'string.escape'],
            [/\n/, 'string', '@pop'],
        ],
        formattedString: [
            {include: '@stringCommon'},
            [/[^\\"{\n]+/, 'string'],
            [/"/, 'string', '@pop'],
            [/\{/, '@brackets', '@interpolation'],
        ],
        interpolation: [[/\{/, '@brackets', '@push'], [/\}/, '@brackets', '@pop'], {include: '@root'}],
    },
});

// Indent after a trailing {, [ or (, allowing whitespace and an optional # comment.
const increaseIndentPattern = /^(?!\s*#).*(?:\{|\[|\()\s*(?:#.*)?$/;
const decreaseIndentPattern = /^\s*[}\])]/;

monaco.languages.setLanguageConfiguration('duckling', {
    comments: {lineComment: '#', blockComment: ['#{', '#}']},
    indentationRules: {
        increaseIndentPattern,
        decreaseIndentPattern,
    },
    onEnterRules: [
        {
            beforeText: increaseIndentPattern,
            afterText: decreaseIndentPattern,
            action: {indentAction: monaco.languages.IndentAction.IndentOutdent},
        },
        {beforeText: increaseIndentPattern, action: {indentAction: monaco.languages.IndentAction.Indent}},
    ],
    brackets: [
        ['{', '}'],
        ['[', ']'],
        ['(', ')'],
    ],
    autoClosingPairs: [
        {open: '{', close: '}'},
        {open: '[', close: ']'},
        {open: '(', close: ')'},
        {open: '"', close: '"', notIn: ['string', 'comment']},
        {open: "'", close: "'", notIn: ['string', 'comment']},
    ],
});

monaco.languages.registerDocumentFormattingEditProvider('duckling', {
    provideDocumentFormattingEdits(model) {
        const configuration = StandaloneServices.get(ILanguageConfigurationService);
        return getReindentEditOperations(model, configuration, 1, model.getLineCount()).map(edit => ({
            range: edit.range,
            text: edit.text ?? '',
        }));
    },
});
