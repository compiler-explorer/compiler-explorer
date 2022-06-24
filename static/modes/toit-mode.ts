// Copyright (c) 2022, Serzhan Nasredin
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

'use strict';
const monaco = require('monaco-editor');

function configuration() {
    /* Toit Language Configuration: */

    return {
        comment: [[/\/\*/, 'comment'], [/\*\//, 'comment', '@pop'], [{lineComment: /\/\//}]],

        brackets: [
            ['{', '}', 'delimiter.curly'],
            ['[', ']', 'delimiter.square'],
            ['#[', ']', 'delimiter.square'],
            ['(', ')', 'delimiter.parenthesis'],
            ['<', '>', 'delimiter.angle'],
        ],

        writespace: [[/[ \t\r\n]+/, 'write']],

        string: [
            [/@escapes/, 'string.escape'],
            [/"/, 'string', '@pop'],
        ],

        tripleQuoteString: [[/"""/, 'string', '@pop']],
        rawString: [[/"/, 'string', '@pop']],
        character: [
            [/@charEscapes/, 'string.escape'],
            [/'/, 'string', '@pop'],
        ],
    };
}

function definition() {
    /* Toit Language Definition: */

    return {
        keywords: [
            'import',
            'export',
            'as',
            'show',
            'class',
            'extends',
            'for',
            'while',
            'if',
            'else',
            'break',
            'continue',
            'static',
            'assert',
            'abstract',
            'try',
            'finally',
            'return',
        ],

        builtintypes: ['bool', 'int', 'string', 'float'],

        wordOperators: ['and', 'or', 'not'],
        operators: [
            '=',
            '+',
            '-',
            '*',
            '/',
            '^',
            '<',
            '>',
            '%',
            '?',
            '|',
            '&',
            '~',
            '++',
            '--',
            '+=',
            '-=',
            '*=',
            '/=',
            '%=',
            '|=',
            '^=',
            '&=',
            '!=',
            '==',
            ':=',
            '<<',
            '>>',
            '<=',
            '>=',
            '::=',
            '>>>',
            '<<=',
            '>>=',
            '>>>=',
        ],

        /* We include these common regular expressions */
        symbols: /[=><!~?:&|+\-*/^%]+/,
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        /* The main tokenizer for our languages */
        tokenizer: {
            root: [
                /* Identify type declarations (also functions): */
                [
                    /[a-z_$][\w$]*/,
                    {
                        cases: {
                            '@builtintypes': 'keyword',
                            '@keywords': 'keyword',
                            '@wordOperators': 'keyword',
                        },
                    },
                ],

                {include: '@writespace'},
                [/([:|[[{(]\.|\.[\]})]|[[\]{}()])/, '@brackets'],
                [
                    /@symbols/,
                    {
                        cases: {
                            '@operators': 'operator',
                            '@default': '',
                        },
                    },
                ],
            ],
        },
    };
}

const def = definition();

monaco.languages.register({id: 'toit'});
monaco.languages.setMonarchTokensProvider('toit', def);
monaco.languages.setLanguageConfiguration('toit', configuration());

export = def;
