// Copyright (c) 2012, Compiler Explorer Authors
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
        ],

        escapes: /\\[abfnrtuvx0-7\\'"`]/,

        symbols: /[+\-*/%^~|#&!.;:<=>?$]*/,

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
                [/\d*\.\d+([e][-+]?\d+)?/, 'number.float'],
                [/0[x][0-9a-fA-F_]*[0-9a-fA-F]/, 'number.hex'],
                [/0o[0-7_]*[0-7]/, 'number.octal'],
                [/0[b][0-1_]*[0-1]/, 'number.binary'],
                [/\d+/, 'number'],

                // delimeter after .\d floats
                [/[;,.]/, 'delimiter- '],

                // strings
                [/'([^'\\]|\\.)*$/, 'string.invalid'], // non-teminated string ('')
                [/'/, 'string', '@single_string'],
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string ("")
                [/"/, 'string', '@double_string'],

                // runes
                [/`[^`]`/, 'string'],
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

            double_string: [
                [/[^\\"]+/, 'string'],
                //  [/@escapes/, 'string.escape'],
                //  [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop'],
            ],

            single_string: [
                [/[^\\']+/, 'string'],
                //    [/@escapes/, 'string.escape'],
                //    [/\\./, 'string.escape.invalid'],
                [/'/, 'string', '@pop'],
            ],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'v'});
monaco.languages.setMonarchTokensProvider('v', def);

export = def;
