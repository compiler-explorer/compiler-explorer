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

function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: 'invalid',

        // the reserved words of mach's lexer
        keywords: [
            'asm',
            'brk',
            'cnt',
            'def',
            'each',
            'error',
            'ext',
            'fin',
            'for',
            'fun',
            'fwd',
            'if',
            'in',
            'or',
            'pub',
            'rec',
            'ret',
            'sel',
            'tag',
            'test',
            'uni',
            'use',
            'val',
            'var',
        ],

        typeKeywords: ['u8', 'u16', 'u32', 'u64', 'i8', 'i16', 'i32', 'i64', 'f32', 'f64', 'ptr'],

        constants: ['nil'],

        decorators: [
            'align',
            'deprecated',
            'embed',
            'inline',
            'library',
            'naked',
            'noinline',
            'oblivious',
            'packed',
            'section',
            'symbol',
        ],

        operators: [
            '::',
            ':~',
            ':>',
            '==',
            '!=',
            '<=',
            '>=',
            '<<',
            '>>',
            '&&',
            '||',
            '...',
            '+',
            '-',
            '*',
            '/',
            '%',
            '^',
            '&',
            '|',
            '=',
            '<',
            '>',
            '!',
            '~',
            '@',
            '?',
        ],

        symbols: /[=><!~?:&|+\-*/^%@.]+/,
        escapes: /\\(?:['"\\tnr0]|x[0-9A-Fa-f]{2})/,

        tokenizer: {
            root: [
                [/(asm)(\s+)([a-zA-Z_]\w*)(\s*)(\{)/, ['keyword', '', 'type', '', {token: '@brackets', next: '@asm'}]],

                [
                    /[a-zA-Z_]\w*/,
                    {
                        cases: {
                            '@keywords': 'keyword',
                            '@typeKeywords': 'type',
                            '@constants': 'constant',
                            '@default': 'identifier',
                        },
                    },
                ],

                [/\$[a-zA-Z_]\w*/, 'keyword'],

                [/#\[/, {token: 'annotation', next: '@decorator'}],

                {include: '@whitespace'},

                [/[{}()[\]]/, '@brackets'],

                [/\d[\d_]*(?:\.\d[\d_]*(?:[eE][+-]?\d[\d_]*)?|[eE][+-]?\d[\d_]*)(?:f\d*)?/, 'number.float'],
                [/0[xX][0-9A-Fa-f_]+(?:[ui]\d*)?/, 'number.hex'],
                [/0[bB][01_]+(?:[ui]\d*)?/, 'number.binary'],
                [/0[oO][0-7_]+(?:[ui]\d*)?/, 'number.octal'],
                [/\d[\d_]*(?:[ui]\d*)?/, 'number'],

                [
                    /@symbols/,
                    {
                        cases: {
                            '@operators': 'operator',
                            '@default': 'delimiter',
                        },
                    },
                ],

                [/[,;]/, 'delimiter'],

                [/"([^"\\]|\\.)*$/, 'string.invalid'],
                [/"/, 'string', '@string'],

                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/#(?!\[).*$/, 'comment'],
            ],

            decorator: [
                [/\]/, {token: 'annotation', next: '@pop'}],
                [/[a-zA-Z_]\w*/, {cases: {'@decorators': 'annotation', '@default': 'identifier'}}],
                [/"/, 'string', '@string'],
                [/\d[\d_]*/, 'number'],
                [/[ \t\r\n]+/, 'white'],
                [/[(),=]/, 'delimiter'],
            ],

            asm: [
                [/\}/, {token: '@brackets', next: '@pop'}],
                [/#.*$/, 'comment'],
                [/\{[a-zA-Z_]\w*\}/, 'variable'],
                [/[^}#{]+/, ''],
                [/\{/, ''],
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

const config: monaco.languages.LanguageConfiguration = {
    comments: {
        lineComment: '#',
    },
    brackets: [
        ['{', '}'],
        ['[', ']'],
        ['(', ')'],
    ],
    autoClosingPairs: [
        {open: '{', close: '}'},
        {open: '[', close: ']'},
        {open: '(', close: ')'},
        {open: '"', close: '"'},
        {open: "'", close: "'"},
    ],
    surroundingPairs: [
        {open: '{', close: '}'},
        {open: '[', close: ']'},
        {open: '(', close: ')'},
        {open: '"', close: '"'},
        {open: "'", close: "'"},
    ],
};

monaco.languages.register({id: 'mach'});
monaco.languages.setMonarchTokensProvider('mach', definition());
monaco.languages.setLanguageConfiguration('mach', config);
