// Copyright (c) 2017, Patrick Quist
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

var monaco = require('../monaco');

function definition() {
    // Object-Pascal language definition

    return {
        keywords: [
            'unit', 'interface', 'implementation', 'uses',
            'function', 'procedure', 'const', 'begin', 'end', 'not', 'while',
            'as', 'for', 'with',
            'else', 'if',
            'break', 'except', 'on',
            'class', 'exec', 'in', 'throw', 'continue', 'finally', 'is',
            'for', 'try', 'then', 'do',
            ':', '=', 'var',
            'strict', 'private', 'protected', 'public', 'published',
            'type'
        ],
        operators: [
            '+', '-', '*', '/', 'div', 'mod',
            'shl', 'shr', 'and', 'or', 'xor', 'not',
            '<', '>', '<=', '>=', '==', '<>',
            '+=', '-=', '*=', '/='
        ],
        brackets: [
            ['(', ')', 'delimiter.parenthesis'],
            ['[', ']', 'delimiter.square']
        ],
        symbols: /[=><!~&|+\-*/^%]+/,
        delimiters: /[;=.@:,`]/,
        escapes: /\\(?:[abfnrtv\\'\n\r]|x[0-9A-Fa-f]{2}|[0-7]{3}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|N\{\w+\})/,
        rawpre: /(?:[rR]|ur|Ur|uR|UR|br|Br|bR|BR)/,
        strpre: /(?:[buBU])/,

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                // strings: need to check first due to the prefix
                [/@strpre?(''')/, {token: 'string.delim', bracket: '@open', next: '@mstring.$1'}],
                [/@strpre?'([^'\\]|\\.)*$/, 'string.invalid'],  // non-teminated string
                [/@strpre?(['])/, {token: 'string.delim', bracket: '@open', next: '@string.$1'}],
                [/@rawpre(''')/, {token: 'string.delim', bracket: '@open', next: '@mrawstring.$1'}],
                [/@rawpre'([^'\\]|\\.)*$/, 'string.invalid'],  // non-teminated string
                [/@rawpre(['])/, {token: 'string.delim', bracket: '@open', next: '@rawstring.$1'}],
                [/__[\w$]*/, 'predefined'],
                [/[a-z_$][\w$]*/, {
                    cases: {
                        '@keywords': 'keyword',
                        '@default': 'identifier'
                    }
                }],
                [/[A-Z][\w]*/, {
                    cases: {
                        '~[A-Z0-9_]+': 'constructor.identifier',
                        '@default': 'namespace.identifier'
                    }
                }],  // to show class names nicely
                {include: '@whitespace'},
                [/[()[\]]/, '@brackets'],
                [/@symbols/, {
                    cases: {
                        '@keywords': 'keyword',
                        '@operators': 'operator',
                        '@default': ''
                    }
                }],
                [/\d*\.\d+([eE][-+]?\d+)?/, 'number.float'],
                [/#$[0-9a-fA-F]+[lL]?/, 'number.hexchar'],
                [/#[bB][0-1]+[lL]?/, 'number.char'],
                [/(0|[1-9]\d*)[lL]?/, 'number'],
                [':', {token: 'keyword', bracket: '@open'}], // bracket for indentation
                [/@delimiters/, {
                    cases: {
                        '@keywords': 'keyword',
                        '@default': 'delimiter'
                    }
                }]
            ],
            comment: [
                [/[^/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'],    // nested comment
                ["\\*/", 'comment', '@pop'],
                [/[/*]/, 'comment']
            ],
            mstring: [
                {include: '@strcontent'},
                [/'''/, {
                    cases: {
                        '$#==$S2': {token: 'string.delim', bracket: '@close', next: '@pop'},
                        '@default': {token: 'string'}
                    }
                }],
                [/["']/, 'string'],
                [/./, 'string.invalid']
            ],
            string: [
                {include: '@strcontent'},
                [/[']/, {
                    cases: {
                        '$#==$S2': {token: 'string.delim', bracket: '@close', next: '@pop'},
                        '@default': {token: 'string'}
                    }
                }],
                [/./, 'string.invalid']
            ],
            strcontent: [
                [/[^\\']+/, 'string'],
                [/\\$/, 'string.escape'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid']
            ],
            mrawstring: [
                {include: '@rawstrcontent'},
                [/'''/, {
                    cases: {
                        '$#==$S2': {token: 'string.delim', bracket: '@close', next: '@pop'},
                        '@default': {token: 'string'}
                    }
                }],
                [/["']/, 'string'],
                [/./, 'string.invalid']
            ],
            rawstring: [
                {include: '@rawstrcontent'},
                [/[']/, {
                    cases: {
                        '$#==$S2': {token: 'string.delim', bracket: '@close', next: '@pop'},
                        '@default': {token: 'string'}
                    }
                }],
                [/./, 'string.invalid']
            ],
            rawstrcontent: [
                [/[^\\']+/, 'string'],
                [/\\[']/, 'string'],
                [/\\/, 'string']
            ],
            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/#.*$/, 'comment']
            ]
        }
    };
}

monaco.languages.register({id: 'pascal'});
monaco.languages.setMonarchTokensProvider('pascal', definition());
