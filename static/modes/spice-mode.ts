// Copyright (c) 2023-2025, Marc Auberer
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
            'alias',
            'alignof',
            'as',
            'assert',
            'break',
            'case',
            'cast',
            'compose',
            'const',
            'continue',
            'default',
            'do',
            'else',
            'enum',
            'ext',
            'f',
            'fallthrough',
            'false',
            'for',
            'foreach',
            'heap',
            'if',
            'import',
            'inline',
            'interface',
            'len',
            'main',
            'nil',
            'operator',
            'p',
            'panic',
            'printf',
            'public',
            'return',
            'signed',
            'sizeof',
            'struct',
            'switch',
            'true',
            'type',
            'unsafe',
            'unsigned',
            'while',
        ],

        typeKeywords: ['double', 'int', 'short', 'long', 'byte', 'char', 'string', 'bool', 'dyn'],

        operators: [
            '+',
            '-',
            '*',
            '/',
            '%',
            '^',
            '~',
            '|',
            '&',
            '++',
            '--',
            '&&',
            '||',
            '!',
            '.',
            '...',
            '::',
            ';',
            ':',
            '=',
            '#',
            '#!',
            '+=',
            '-=',
            '*=',
            '/=',
            '%=',
            '|=',
            '&=',
            '^=',
            '>>=',
            '<<=',
            '==',
            '!=',
            '>',
            '<',
            '>=',
            '<=',
            '?',
            '<<',
            '>>',
            '->',
        ],

        symbols: /[=><!~?:&|+\-*/^%]+/,

        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        tokenizer: {
            root: [
                // identifiers and keywords
                [
                    /[a-z_][a-zA-Z0-9_]*/,
                    {
                        cases: {
                            '@typeKeywords': 'keyword',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],

                [/[A-Z][a-zA-Z0-9_]*/, 'type.identifier'], // to show class names nicely

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
                [/[0-9]*[.][0-9]+([eE][+-]?[0-9]+)?/, 'number.float'],
                [/0[xXhH][0-9a-fA-F]+[sl]?/, 'number.hex'],
                [/0[oO][0-7]+[sl]?/, 'number.octal'],
                [/0[bB][01][sl]?/, 'number.binary'],
                [/(0[dD])?[0-9]+[sl]?/, 'number'],

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // double-quoted strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                [/c?\\\\.*$/, 'string'],
                [/c?"/, 'string', '@double_quoted_string'],

                // characters
                [/'[^\\']+'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            whitespace: [
                [/[ \r\n]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\+/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment'],
                [/\t/, 'comment.invalid'],
            ],

            comment: [
                [/[^/*]+/, 'comment'],
                [/\/\*/, 'comment', '@comment'],
                [/\*\//, 'comment', '@pop'],
                [/[/*]/, 'comment'],
            ],

            characters: [
                [/'[^\\']+'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            double_quoted_string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'spice'});
monaco.languages.setMonarchTokensProvider('spice', def);

export default def;
