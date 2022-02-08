// Copyright (c) 2021, Compiler Explorer Authors
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

const monaco = require('monaco-editor');

function definition() {
    return {
        commands: ['-module', '-export', '-compile', '-record'],

        funcdef: ['when', '->', 'if', 'end', 'unknown', 'case', 'of', 'receive', 'after'],

        operators: [
            '<=', '>=', '==', '!=', '=<', '+', '-', '*', '/',
        ],

        symbols: /[=><!+\-*/]+/,

        escapes: /~(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        tokenizer: {
            root: [
                [/-?\d[\d.]*/, 'number'],

                [/-[a-zA-Z][\w]*/, {
                    cases: {
                        '@commands': 'keyword',
                        '@default': '',
                    },
                }],

                [/[a-zA-Z-][>\w]*/, {
                    cases: {
                        '@funcdef': 'keyword',
                        '@default': 'identifier',
                    },
                }],

                [/[(){}[\]]/, '@brackets'],
                [/<<.*>>/, '@brackets'],

                [/@symbols/, {
                    cases: {
                        '@operators': 'delimiter',
                        '@default': '',
                    },
                }],

                [/^%.*/, 'comment'],

                [/"/, 'string', '@stringDouble'],
                [/"[^\\"]"/, 'string'],
                [/(")(@escapes)(")/, ['string', 'string.escape', 'string']],

                [/'/, 'string', '@stringSingle'],
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],

                [/[;.,]/, 'delimiter'],
            ],

            whitespace: [
                [/\s/],
            ],

            comment: [
                [/%/, 'comment'],
            ],

            stringDouble: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/"/, 'string', '@pop'],
            ],

            stringSingle: [
                [/[^\\']+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/'/, 'string', '@pop'],
            ],
        },
    };
}

function configuration() {
    return {
        comments: {
            lineComment: '%',
        },
        brackets: [
            ['{', '}'],
            ['[', ']'],
            ['(', ')'],
            ['<<', '>>'],
        ],
        autoClosingPairs: [
            { open: '{', close: '}', notIn: ['string', 'comment'] },
            { open: '[', close: ']', notIn: ['string', 'comment'] },
            { open: '(', close: ')', notIn: ['string', 'comment'] },
            { open: '<<', close: '>>', notIn: ['string', 'comment'] },
            { open: "'", close: "'", notIn: ['string', 'comment'] },
            { open: '"', close: '"' },
        ],
        folding: {
            markers: {
                start: '^\\s*\\%\\%region\\b',
                end: '^\\s*\\%\\%endregion\\b',
            },
        },
    };
}

const def = definition();

monaco.languages.register({ id: 'erlang' });
monaco.languages.setMonarchTokensProvider('erlang', def);
monaco.languages.setLanguageConfiguration('erlang', configuration());

export = def;
