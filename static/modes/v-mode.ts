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

function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: 'invalid',

        keywords: [
            '__global',
            '_likely_',
            '_unlikely_',
            'as',
            'asm',
            'assert',
            'atomic',
            'break',
            'const',
            'continue',
            'defer',
            'dump',
            'else',
            'enum',
            'false',
            'fn',
            'for',
            'go',
            'goto',
            'if',
            'import',
            'in',
            'interface',
            'is',
            'isreftype',
            'it',
            'like',
            'lock',
            'match',
            'module',
            'mut',
            'nil',
            'none',
            'offsetof',
            'or',
            'pub',
            'return',
            'rlock',
            'select',
            'shared',
            'sizeof',
            'spawn',
            'static',
            'struct',
            'true',
            'type',
            'typeof',
            'union',
            'unsafe',
            'volatile',
        ],

        typeKeywords: [
            'i8',
            'u8',
            'i16',
            'u16',
            'int',
            'u32',
            'i64',
            'u64',
            'f32',
            'f64',
            'string',
            'map',
            'struct',
            'bool',
            'voidptr',
            'charptr',
            'isize',
            'usize',
        ],

        operators: [
            '+',
            '-',
            '*',
            '/',
            '%',
            '^',
            '~',
            '|',
            '#',
            '&',
            '++',
            '--',
            '&&',
            '||',
            '!',
            '.',
            '!in',
            '!is',
            ';',
            ':',
            '<-',
            '=',
            ':=',
            '+=',
            '-=',
            '*=',
            '/=',
            '%=',
            '|=',
            '&=',
            '>>=',
            '>>>=',
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
            '>>>',
            '$',
        ],

        symbols: /[=><!~?:&|+\-*/^%]+/,

        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

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

                [/@[a-zA-Z_$]*/, 'builtin.identifier'],

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
                [/[0-9_]*\.[0-9_]+([eE][-+]?[0-9_]+)?/, 'number.float'],
                [/0[xX][0-9a-fA-F_]*[0-9a-fA-F_]/, 'number.hex'],
                [/0o[0-7_]*[0-7_]/, 'number.octal'],
                [/0[bB][0-1_]*[0-1_]/, 'number.binary'],
                [/[0-9_]+/, 'number'],

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // single-quoted strings
                [/'([^'\\]|\\.)*$/, 'string.invalid'],
                [/c?\\\\.*$/, 'string'],
                [/c?'/, 'string', '@single_quoted_string'],

                // double-quoted strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                [/c?\\\\.*$/, 'string'],
                [/c?"/, 'string', '@double_quoted_string'],

                // runes
                [/`[^\\`]`/, 'string'],
                [/(`)(@escapes)(`)/, ['string', 'string.escape', 'string']],
                [/`/, 'string.invalid'],
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

            single_quoted_string: [
                [/[^\\']+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/'/, 'string', '@pop'],
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
monaco.languages.register({id: 'v'});
monaco.languages.setMonarchTokensProvider('v', def);

export = def;
