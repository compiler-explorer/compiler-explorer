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

monaco.languages.register({id: 'duckling'});

monaco.languages.setMonarchTokensProvider('duckling', {
    keywords: [
        'and',
        'assert',
        'block',
        'box',
        'break',
        'case',
        'catch',
        'class',
        'compile_assert',
        'const',
        'continue',
        'copy',
        'copyof',
        'cptr',
        'debug',
        'defer',
        'destroy',
        'else',
        'expand',
        'export',
        'extends',
        'extern',
        'false',
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
        'manyptr',
        'match',
        'move',
        'namespace',
        'new',
        'not',
        'or',
        'pattern',
        'private',
        'protected',
        'ptr',
        'ptrof',
        'public',
        'redo',
        'ref',
        'refof',
        'restart',
        'return',
        'self',
        'sizeof',
        'slice',
        'switch',
        'template',
        'then',
        'throw',
        'true',
        'try',
        'using',
        'var',
        'while',
        'with',
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
    operators: [
        '->',
        '=>',
        '==',
        '!=',
        '<=',
        '>=',
        '&&',
        '||',
        '+',
        '-',
        '*',
        '/',
        '%',
        '=',
        '<',
        '>',
        '!',
        '&',
        '|',
        '^',
        '~',
        '?',
        ':',
    ],
    operatorChar: /[=><!~?:&|+\-*/^%.\\`]/,
    numericSuffix: /(?:[iu](?:8|16|32|64|128)|f(?:16|32|64|128))\b/,
    symbols: /[=><!~?:&|+\-*/^%]+/,
    tokenizer: {
        root: [
            [/#\{/, 'comment', '@blockComment'],
            [/#.*$/, 'comment'],
            {include: '@whitespace'},
            [
                /[a-zA-Z_][\w]*/,
                {cases: {'@typeKeywords': 'type.identifier', '@keywords': 'keyword', '@default': 'identifier'}},
            ],
            [/0[xX][0-9a-fA-F]+@numericSuffix?/, 'number.hex'],
            [/0[bB][01]+@numericSuffix?/, 'number.binary'],
            [/0[oO][0-7]+@numericSuffix?/, 'number.octal'],
            [/(?:\d+(?:\.(?!@operatorChar)\d*)?|\.\d+)(?:[eE][+-]?\d+)?@numericSuffix?/, 'number'],
            [/"/, 'string', '@doubleString'],
            [/'/, 'string', '@singleString'],
            [/@symbols/, {cases: {'@operators': 'operator', '@default': ''}}],
            [/[{}()[\]]/, '@brackets'],
            [/[;,.]/, 'delimiter'],
        ],
        whitespace: [[/[ \t\r\n]+/, 'white']],
        blockComment: [
            [/#\}/, 'comment', '@pop'],
            [/[^#]+/, 'comment'],
            [/#/, 'comment'],
        ],
        doubleString: [
            [/[^\\"]+/, 'string'],
            [/\\./, 'string.escape'],
            [/"/, 'string', '@pop'],
        ],
        singleString: [
            [/[^\\']+/, 'string'],
            [/\\./, 'string.escape'],
            [/'/, 'string', '@pop'],
        ],
    },
});

monaco.languages.setLanguageConfiguration('duckling', {
    comments: {lineComment: '#', blockComment: ['#{', '#}']},
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
