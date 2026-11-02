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

monaco.languages.register({id: 'gleam'});

monaco.languages.setMonarchTokensProvider('gleam', {
    keywords: [
        'as',
        'assert',
        'case',
        'const',
        'echo',
        'fn',
        'if',
        'import',
        'let',
        'opaque',
        'panic',
        'pub',
        'todo',
        'type',
        'use',
    ],
    operators: [
        '+',
        '-',
        '*',
        '/',
        '%',
        '+.',
        '-.',
        '*.',
        '/.',
        '<',
        '>',
        '<=',
        '>=',
        '<.',
        '>.',
        '<=.',
        '>=.',
        '==',
        '!=',
        '<>',
        '|>',
        '->',
        '<-',
        '..',
        '<<',
        '>>',
        '=',
        '|',
        '&&',
        '||',
    ],
    symbols: /[=><!+\-*/|.&%]+/,
    tokenizer: {
        root: [
            [/\/\/.*$/, 'comment'],
            [/@[a-z_][\w]*/, 'keyword'],
            [/[A-Z][\w]*/, 'type.identifier'],
            [/[a-z_][\w]*/, {cases: {'@keywords': 'keyword', '@default': 'identifier'}}],
            [/0[bB][01](?:_?[01])*/, 'number.binary'],
            [/0[oO][0-7](?:_?[0-7])*/, 'number.octal'],
            [/0[xX][0-9a-fA-F](?:_?[0-9a-fA-F])*/, 'number.hex'],
            [/\d(?:_?\d)*(?:(?:\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?|[eE][+-]?\d(?:_?\d)*)/, 'number.float'],
            [/\d(?:_?\d)*/, 'number'],
            [/"/, 'string', '@string'],
            [/@symbols/, {cases: {'@operators': 'operator', '@default': ''}}],
            [/[{}()[\]]/, '@brackets'],
            [/[#,:;]/, 'delimiter'],
        ],
        string: [
            [/[^\\"]+/, 'string'],
            [/\\(?:["\\fnrt]|u\{[0-9a-fA-F]{1,6}\})/, 'string.escape'],
            [/\\./, 'string.escape.invalid'],
            [/"/, 'string', '@pop'],
        ],
    },
});

monaco.languages.setLanguageConfiguration('gleam', {
    comments: {lineComment: '//'},
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
    ],
});
