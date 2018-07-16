// Copyright (c) 2017, Matt Godbolt
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


"use strict";

function definition() {
    return {
        defaultToken: 'invalid',

        keywords: [
            'abstract', 'alignof', 'as', 'become', 'box',
            'break', 'const', 'continue', 'crate', 'do',
            'else', 'enum', 'extern', 'false', 'final',
            'fn', 'for', 'if', 'impl', 'in',
            'let', 'loop', 'macro', 'match', 'mod',
            'move', 'mut', 'offsetof', 'override', 'priv',
            'proc', 'pub', 'pure', 'ref', 'return',
            'Self', 'self', 'sizeof', 'static', 'struct',
            'super', 'trait', 'true', 'type', 'typeof',
            'unsafe', 'unsized', 'use', 'virtual', 'where',
            'while', 'yield'
        ],

        typeKeywords: [
            'array', 'bool', 'char', 'f32', 'f64', 'i16', 'i32', 'i64', 'i8',
            'isize', 'pointer', 'slice', 'str', 'tuple', 'u16', 'u32', 'u64', 'u8',
            'usize'
        ],

        operators: [
            '=', '>', '<', '!', '~', '?', ':',
            '==', '<=', '>=', '!=', '&&', '||', '++', '--',
            '+', '-', '*', '/', '&', '|', '^', '%', '<<',
            '>>', '>>>', '+=', '-=', '*=', '/=', '&=', '|=',
            '^=', '%=', '<<=', '>>=', '>>>='
        ],

        symbols: /[=><!~?:&|+\-*/^%]+/,
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                // identifiers and keywords
                [/[a-z_$][\w$]*/, {
                    cases: {
                        '@typeKeywords': 'keyword',
                        '@keywords': 'keyword',
                        '@default': 'identifier'
                    }
                }],
                [/[A-Z][\w$]*/, 'type.identifier'],  // to show class names nicely

                // whitespace
                {include: '@whitespace'},

                // delimiters and operators
                [/[{}()[\]]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],
                [/@symbols/, {
                    cases: {
                        '@operators': 'operator',
                        '@default': ''
                    }
                }],

                [/#!\[[^]*]/, 'annotation'],
                [/#!.*$/, 'annotation.invalid'],

                // numbers
                [/\d*\.\d+([eE][-+]?\d+)?[fFdD]?/, 'number.float'],
                [/0[xX][0-9a-fA-F_]*[0-9a-fA-F][Ll]?/, 'number.hex'],
                [/0[0-7_]*[0-7][Ll]?/, 'number.octal'],
                [/0[bB][0-1_]*[0-1][Ll]?/, 'number.binary'],
                [/\d+[lL]?/, 'number'],

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'],  // non-teminated string
                [/"/, 'string', '@string'],

                // characters
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid']
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment']
            ],

            comment: [
                [/[^/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'],
                [/\/\*/, 'comment.invalid'],
                ["\\*/", 'comment', '@pop'],
                [/[/*]/, 'comment']
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop']
            ]
        }
    };
}

monaco.languages.register({id: 'rust'});
monaco.languages.setMonarchTokensProvider('rust', definition());
