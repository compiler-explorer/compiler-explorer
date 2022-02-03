// Copyright (c) 2018, Marc Tiehuis
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

function definition() {
    return {
        defaultToken: 'invalid',

        keywords: [
            'const', 'var', 'extern', 'packed', 'export', 'pub', 'noalias', 'inline',
            'comptime', 'nakedcc', 'stdcallcc', 'volatile', 'align', 'section', 'linksection',
            'struct', 'enum', 'union',
            'break', 'return', 'continue', 'asm', 'defer', 'errdefer', 'unreachable',
            'try', 'catch', 'async', 'await', 'suspend', 'resume', 'cancel',
            'if', 'else', 'switch', 'and', 'or', 'orelse',
            'while', 'for',
            'null', 'undefined',
            'fn', 'usingnamespace', 'use', 'test',
            'this',
        ],
        typeKeywords: [
            'bool', 'f32', 'f64', 'f128', 'void', 'noreturn', 'type', 'error', 'anyerror', 'promise', 'anyframe',
            'isize', 'usize', 'c_short', 'c_ushort', 'c_int', 'c_uint', 'c_long', 'c_ulong',
            'c_longlong', 'c_ulonglong', 'c_longdouble', 'c_void',
        ],
        operators: [
            '+', '+%', '-', '-%', '/', '*', '*%', '=', '^', '&', '?', '|',
            '!', '>', '<', '%', '<<', '<<%', '>>',
            '+=', '+%=', '-=', '-%=', '/=', '*=', '*%=', '==', '^=', '&=',
            '?=', '|=', '!=', '>=', '<=', '%=', '<<=', '<<%=', '>>=',
        ],

        symbols: /[=><!~?:&|+\-*/^%]+/,

        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        tokenizer: {
            root: [
                // u0/i0 integer types
                [/[iu]\d+/, 'keyword'],

                // identifiers and keywords
                [/[a-z_$][\w$]*/, {
                    cases: {
                        '@typeKeywords': 'keyword',
                        '@keywords': 'keyword',
                        '@default': 'identifier',
                    },
                }],

                [/@[a-zA-Z_$]*/, 'builtin.identifier'],

                [/[A-Z][\w$]*/, 'type.identifier'],  // to show class names nicely

                // whitespace
                {include: '@whitespace'},

                // delimiters and operators
                [/[{}()[\]]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],
                [/@symbols/, {
                    cases: {
                        '@operators': 'operator',
                        '@default': '',
                    },
                }],

                // numbers
                [/\d*\.\d+([eE][-+]?\d+)?[fFdD]?/, 'number.float'],
                [/0[xX][0-9a-fA-F_]*[0-9a-fA-F][Ll]?/, 'number.hex'],
                [/0o[0-7_]*[0-7][Ll]?/, 'number.octal'],
                [/0[bB][0-1_]*[0-1][Ll]?/, 'number.binary'],
                [/\d+/, 'number'],

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'],  // non-teminated string
                [/c?\\\\.*$/, 'string'],
                [/c?"/, 'string', '@string'],

                // characters
                [/'[^\\']'/, 'string'],
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
                [/\/\*/, 'comment.invalid'],
                [/[/*]/, 'comment'],
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

monaco.languages.register({id: 'zig'});
monaco.languages.setMonarchTokensProvider('zig', definition());

export {};
