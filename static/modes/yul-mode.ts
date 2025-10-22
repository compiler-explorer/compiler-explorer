// Copyright (c) 2025, Compiler Explorer Authors
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

export function definition(): monaco.languages.IMonarchLanguage {
    return {
        // Set defaultToken to 'invalid' to see what is not yet tokenized.
        defaultToken: 'invalid',

        keywords: [
            'break',
            'case',
            'code',
            'continue',
            'data',
            'default',
            'false',
            'for',
            'function',
            'hex',
            'if',
            'leave',
            'let',
            'object',
            'switch',
            'true',
        ],

        operators: [':='],

        brackets: [
            {open: '{', close: '}', token: 'delimiter.curly'},
            {open: '(', close: ')', token: 'delimiter.parenthesis'},
            {open: '[', close: ']', token: 'delimiter.square'},
        ],

        symbols: /[:=]+/,

        escapes: /\\(?:['"\\nrt\n\r]|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4})/,

        tokenizer: {
            root: [
                // Identifiers and keywords
                [
                    /[a-z][a-zA-Z0-9_]*/,
                    {
                        cases: {
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],

                [/[a-zA-Z_$][a-zA-Z_$0-9.]*/, 'type.identifier'],

                // Whitespace
                {include: '@whitespace'},

                // Delimiters and operators
                [/[()[\]{}]/, '@brackets'],
                [/[.,:]/, 'delimiter'],

                [
                    /@symbols/,
                    {
                        cases: {
                            '@operators': 'operator',
                            '@default': '',
                        },
                    },
                ],

                // Numbers
                [/0x[0-9a-fA-F]+/, 'number.hex'],
                [/\d+/, 'number'],

                // Strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-terminated string
                [/"/, 'string', '@string'],
            ],

            // Whitespace and comments
            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment'],
            ],

            comment: [
                [/[^/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'], // Nested comment
                ['\\*/', 'comment', '@pop'],
                [/[/*]/, 'comment'],
            ],

            // Strings
            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

function configuration(): monaco.languages.LanguageConfiguration {
    return {
        comments: {
            lineComment: '//',
            blockComment: ['/*', '*/'],
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
            {open: '"', close: '"', notIn: ['string']},
            {open: '/*', close: ' */', notIn: ['string']},
        ],

        surroundingPairs: [
            {open: '{', close: '}'},
            {open: '[', close: ']'},
            {open: '(', close: ')'},
            {open: '"', close: '"'},
        ],
    };
}

monaco.languages.register({id: 'yul'});
monaco.languages.setMonarchTokensProvider('yul', definition());
monaco.languages.setLanguageConfiguration('yul', configuration());
