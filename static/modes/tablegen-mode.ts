// Copyright (c) 2023, Compiler Explorer Authors
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
        keywords: [
            'assert',
            'class',
            'code',
            'def',
            'dump',
            'else',
            'false',
            'foreach',
            'defm',
            'defset',
            'defvar',
            'field',
            'if',
            'in',
            'include',
            'let',
            'multiclass',
            'then',
            'true',
        ],
        standardTypes: ['bit', 'int', 'string', 'dag', 'bits', 'list'],
        operators: [
            '!add',
            '!and',
            '!cast',
            '!con',
            '!cond',
            '!dag',
            '!div',
            '!empty',
            '!eq',
            '!exists',
            '!filter',
            '!find',
            '!foldl',
            '!foreach',
            '!ge',
            '!getdagarg',
            '!getdagname',
            '!getdagop',
            '!gt',
            '!head',
            '!if',
            '!interleave',
            '!isa',
            '!le',
            '!listconcat',
            '!listremove',
            '!listsplat',
            '!logtwo',
            '!lt',
            '!mul',
            '!ne',
            '!not',
            '!or',
            '!range',
            '!repr',
            '!setdagarg',
            '!setdagname',
            '!setdagop',
            '!shl',
            '!size',
            '!sra',
            '!srl',
            '!strconcat',
            '!sub',
            '!subst',
            '!substr',
            '!tail',
            '!tolower',
            '!toupper',
            '!xor',
        ],
        brackets: [
            {
                open: '(',
                close: ')',
                token: 'delimiter.parenthesis',
            },
            {
                open: '[',
                close: ']',
                token: 'delimiter.square',
            },
            {
                open: '<',
                close: '>',
                token: 'delimiter.angle',
            },
        ],
        symbols: /[=><!~&|+\-*/^]+/,
        delimiters: /[;=.:,`]/,
        escapes: /\\(?:[abfnrtv\\'\n\r]|x[0-9A-Fa-f]{2}|[0-7]{3}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|N\{\w+\})/,

        tokenizer: {
            root: [
                [
                    /[a-zA-Z_][a-zA-Z0-9_]*/,
                    {
                        cases: {
                            '@standardTypes': 'type',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],
                {include: '@whitespace'},

                [/[()[\]<>]/, '@brackets'],

                // Numbers
                [/0x([abcdef]|[ABCDEF]|\d)+/, 'number.hex'],
                [/0b[01]+1/, 'number.binary'],
                // Decimal may have + or - in front.
                [/[-+]?\d+/, 'number'],

                // Strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                [/"/, 'string', '@string'],

                [
                    /@delimiters/,
                    {
                        cases: {
                            '@keywords': 'keyword',
                            '@default': 'delimiter',
                        },
                    },
                ],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\+/, 'comment', '@nestingcomment'],
                [/\/\/.*$/, 'comment'],
            ],

            comment: [
                [/[^/*]+/, 'comment'],
                [/\*\//, 'comment', '@pop'],
                [/[/*]/, 'comment'],
            ],

            nestingcomment: [
                [/[^/+]+/, 'comment'],
                [/\/\+/, 'comment', '@push'],
                [/\/\+/, 'comment.invalid'],
                [/\+\//, 'comment', '@pop'],
                [/[/+]/, 'comment'],
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
monaco.languages.register({id: 'tablegen'});
monaco.languages.setMonarchTokensProvider('tablegen', definition());

export {};
