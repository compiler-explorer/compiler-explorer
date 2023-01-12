// Copyright (c) 2022, Compiler Explorer Authors
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
const cpp = require('monaco-editor/esm/vs/basic-languages/cpp/cpp');

function definition() {
    return {
        defaultToken: 'invalid', // for debugging

        keywords: [
            'abstract',
            'addr',
            'alias',
            'and',
            'api',
            'as',
            'auto',
            '__await',
            'base',
            'break',
            'case',
            'choice',
            'class',
            '__continuation',
            'continue',
            'default',
            'destructor',
            'else',
            'extends',
            'external',
            'false',
            'final',
            'forall',
            'fn',
            'if',
            'impl',
            'import',
            'interface',
            'let',
            'library',
            'match',
            'namespace',
            'not',
            'or',
            'package',
            'private',
            'protected',
            'return',
            'returned',
            '__run',
            'template',
            'then',
            'true',
            'var',
            'virtual',
            'while',
        ],

        typeKeywords: [
            'bool',
            'Carbon.Int',
            'Carbon.UInt',
            '__Continuation',
            'f16',
            'f32',
            'f64',
            'f128',
            '__Fn',
            'i8',
            'i16',
            'i32',
            'i64',
            'i128',
            'i256',
            'Slice',
            'String',
            'StringView',
            'Type',
            'u8',
            'u16',
            'u32',
            'u64',
            'u128',
            'u256',
        ],

        operators: [
            '&',
            '->',
            ':',
            ':!',
            ',',
            '=>',
            '=',
            '==',
            '-',
            '.',
            '+',
            ';',
            '/',
            '_', // not sure _ is an operator really
        ],

        // we include these common regular expressions
        symbols: /[=><!~?:&|+\-*/^%]+/,

        // C# style strings (TODO)
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        // The main tokenizer for our languages
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
                [/\d*\.\d+([eE][-+]?\d+)?/, 'number.float'],
                [/0[xX][0-9a-fA-F]+/, 'number.hex'],
                [/\d+/, 'number'],

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-terminated string
                [/"/, {token: 'string.quote', bracket: '@open', next: '@string'}],

                // characters
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            comment: [
                [/[^/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'], // nested comment
                ['\\*/', 'comment', '@pop'],
                [/[/*]/, 'comment'],
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, {token: 'string.quote', bracket: '@close', next: '@pop'}],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment'],
            ],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'carbon'});
monaco.languages.setMonarchTokensProvider('carbon', def);
monaco.languages.setLanguageConfiguration('carbon', cpp.conf);

export = def;
