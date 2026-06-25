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

        keywords: [
            'access',
            'at',
            'begin',
            'by',
            'case',
            'co',
            'comment',
            'def',
            'do',
            'egg',
            'elif',
            'else',
            'empty',
            'end',
            'esac',
            'exit',
            'false',
            'fed',
            'fi',
            'for',
            'format',
            'from',
            'go',
            'goto',
            'hole',
            'if',
            'in',
            'is',
            'isnt',
            'mode',
            'module',
            'nest',
            'nil',
            'od',
            'of',
            'op',
            'ouse',
            'out',
            'par',
            'pr',
            'pragmat',
            'prio',
            'proc',
            'pub',
            'skip',
            'struct',
            'then',
            'to',
            'true',
            'union',
            'while',
        ],

        typeKeywords: [
            'bits',
            'bool',
            'bytes',
            'channel',
            'char',
            'compl',
            'file',
            'flex',
            'heap',
            'int',
            'loc',
            'long',
            'real',
            'ref',
            'sema',
            'short',
            'string',
            'void',
        ],

        operators: ['+', '-', '/', '%', '%*', '^', '&'],

        symbols: /(?:[%^&+\-~!?></=*]|[%^&+\-~!?][></=*]?(?::=|=:)?|:=|=:)/,
        escapes: /(?:'(?:[fnrt']|\((?:u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})\))|"")/,

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                // tags and reserved words
                [
                    /[a-z]_?(?:[a-z0-9]+_?)*/,
                    {
                        cases: {
                            '@typeKeywords': 'keyword',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],
                [/[A-Z]_?(?:[A-Za-z0-9]+_?)*/, 'type.identifier'], // to show bold words differently than tags

                // whitespace
                {include: '@whitespace'},

                // numbers
                [/\d*\.\d+([eE][-+]?\d+)?/, 'number.float'],
                [/16r[0-9a-fA-F][ 0-9a-fA-F]*/, 'number.hex'],
                [/8r[0-7][ 0-7]*/, 'number.octal'],
                [/2r[0-1][ 0-1]*/, 'number.binary'],
                [/\d+/, 'number'],

                // delimiters and operators
                [/[()[\]]/, '@brackets'],
                [/@symbols/, 'operator'],
                [/[;,]/, 'delimiter'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                [/"/, 'string', '@string'],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/{/, 'comment', '@nestingcomment'],
            ],

            nestingcomment: [
                [/[^{}]+/, 'comment'],
                [/{/, 'comment', '@push'],
                [/}/, 'comment', '@pop'],
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

function configuration(): monaco.languages.LanguageConfiguration {
    return {
        comments: {
            blockComment: ['{', '}'],
        },

        brackets: [
            ['[', ']'],
            ['(', ')'],
        ],

        autoClosingPairs: [
            {open: '{', close: '}'},
            {open: '[', close: ']'},
            {open: '(', close: ')'},
            {open: '"', close: '"', notIn: ['string']},
        ],

        surroundingPairs: [
            {open: '{', close: '}'},
            {open: '[', close: ']'},
            {open: '(', close: ')'},
            {open: '"', close: '"'},
        ],
    };
}

monaco.languages.register({id: 'algol68'});
monaco.languages.setMonarchTokensProvider('algol68', definition());
monaco.languages.setLanguageConfiguration('algol68', configuration());
