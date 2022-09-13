// Copyright (c) 2020, Lock3 Software LLC
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

// Originally based on `./d-mode.js` by the Compiler Explorer Authors

'use strict';
const monaco = require('monaco-editor');

function definition() {
    return {
        defaultToken: '',

        brackets: [
            {token: 'delimiter.curly', open: '{', close: '}'},
            {token: 'delimiter.parenthesis', open: '(', close: ')'},
            {token: 'delimiter.square', open: '[', close: ']'},
            {token: 'delimiter.angle', open: '<', close: '>'},
        ],

        keywords: [
            'array',
            'auto',
            'bool',
            'break',
            'case',
            'catch',
            'char',
            'class',
            'const',
            'constexpr',
            'const_cast',
            'continue',
            'decltype',
            'default',
            'delete',
            'do',
            'dynamic_cast',
            'else',
            'enum',
            'explicit',
            'export',
            'extern',
            'false',
            'final',
            'for',
            'if',
            'in',
            'inline',
            'mutable',
            'namespace',
            'new',
            'noexcept',
            'operator',
            'override',
            'private',
            'protected',
            'public',
            'register',
            'reinterpret_cast',
            'return',
            'sizeof',
            'static',
            'static_assert',
            'static_cast',
            'switch',
            'template',
            'this',
            'thread_local',
            'throw',
            'tile_static',
            'true',
            'try',
            'typedef',
            'typeid',
            'typename',
            'union',
            'using',
            'virtual',
            'void',
            'volatile',
            'wchar_t',
            'where',
            'while',

            // Additional C++ keywords
            'alignas',
            'alignof',
            'and',
            'or',
            'not',

            // Gold specific keywords
            'returns',
            'otherwise',
            'then',
            'until',
            'null',
            'ref',
            'rref',
        ],

        typeKeywords: [
            // C++ keywords
            'int',
            'double',
            'float',

            // Gold specific keywords
            'type',
            'uint',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
            'uint128',
            'float16',
            'float32',
            'float64',
            'float128',
            'int8',
            'int16',
            'int32',
            'int64',
            'int128',
            'type',
            'char8',
            'char16',
            'char32',
            'null_t',
        ],

        operators: [
            '=',
            '>',
            '<',
            '!',
            '~',
            '?',
            ':',
            '==',
            '<=',
            '>=',
            '<>',
            '&&',
            '||',
            '+',
            '-',
            '*',
            '/',
            '&',
            '|',
            '^',
            '%',
            '<<',
            '>>',
            '>>>',
            '+=',
            '-=',
            '*=',
            '/=',
            '&=',
            '|=',
            '^=',
            '%=',
            '<<=',
            '>>=',
            '>>>=',
        ],

        symbols: /[=><!~?:&|+\-*/^%]+/,
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        // The main tokenizer
        tokenizer: {
            root: [
                // identifiers and keywords
                [
                    /[a-z_$][\w$]*/,
                    {
                        cases: {
                            '@typeKeywords': 'keyword',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],
                [/[A-Z][\w$]*/, 'type.identifier'], // to show class names nicely

                // whitespace
                {include: '@whitespace'},

                // delimiters and operators
                [/[{}()[\]]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],
                [
                    /@symbols/,
                    {
                        cases: {
                            '@operators': 'operator',
                            '@default': '',
                        },
                    },
                ],

                // numbers
                [/\d*\.\d+([eE][-+]?\d+)?[fFdD]?/, 'number.float'],
                [/0[xX][0-9a-fA-F_]*[0-9a-fA-F][Ll]?/, 'number.hex'],
                [/0[0-7_]*[0-7][Ll]?/, 'number.octal'],
                [/0[bB][0-1_]*[0-1][Ll]?/, 'number.binary'],
                [/\d+[lL]?/, 'number'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                [/"/, 'string', '@string'],

                // characters
                [/'[^\\']+'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            // strings
            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/"/, 'string', '@pop'],
            ],

            // characters
            characters: [
                [/'[^\\']+'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/<#/, 'comment', '@nestingcomment'],
                [/#.*$/, 'comment'],
            ],

            comment: [[/[#]/, 'comment']],

            nestingcomment: [
                [/[^<#]+/, 'comment'],
                [/<#/, 'comment', '@push'],
                [/<#/, 'comment.invalid'],
                [/#>/, 'comment', '@pop'],
                [/[<#]/, 'comment'],
            ],
        },
    };
}

function configuration() {
    return {
        comments: {
            lineComment: '#',
            blockComment: ['<#', '#>'],
        },

        brackets: [
            ['{', '}'],
            ['[', ']'],
            ['(', ')'],
        ],

        autoClosingPairs: [
            {open: '[', close: ']'},
            {open: '{', close: '}'},
            {open: '(', close: ')'},
            {open: "'", close: "'", notIn: ['string', 'comment']},
            {open: '"', close: '"', notIn: ['string']},
        ],

        surroundingPairs: [
            {open: '{', close: '}'},
            {open: '[', close: ']'},
            {open: '(', close: ')'},
            {open: '"', close: '"'},
            {open: "'", close: "'"},
        ],
    };
}

monaco.languages.register({id: 'cppx-gold'});
monaco.languages.setMonarchTokensProvider('cppx-gold', definition());
monaco.languages.setLanguageConfiguration('cppx-gold', configuration());

export {};
