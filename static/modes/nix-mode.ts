// Copyright (c) 2025, Compiler Explorer Authors
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
        defaultToken: '',
        tokenPostfix: '.nix',

        // Keywords in Nix
        keywords: ['if', 'then', 'else', 'with', 'assert', 'let', 'in', 'rec', 'inherit'],

        // Built-in constants and functions
        builtins: ['builtins', 'true', 'false', 'null'],
        functions: [
            'scopedImport',
            'import',
            'isNull',
            'abort',
            'throw',
            'baseNameOf',
            'dirOf',
            'removeAttrs',
            'map',
            'toString',
            'derivationStrict',
            'derivation',
        ],

        // Operators and punctuation
        operators: [
            'or',
            '.',
            '|>',
            '<|',
            '==',
            '!=',
            '!',
            '<=',
            '<',
            '>=',
            '>',
            '&&',
            '||',
            '->',
            '//',
            '?',
            '++',
            '-',
            '*',
            '/',
            '+',
        ],
        brackets: [
            {open: '{', close: '}', token: 'delimiter.curly'},
            {open: '[', close: ']', token: 'delimiter.square'},
            {open: '(', close: ')', token: 'delimiter.parenthesis'},
        ],

        // Tokenizer rules
        tokenizer: {
            root: [
                // Whitespace
                {include: '@whitespace'},

                // Comments
                [/#.*$/, 'comment'],
                [/\/\*([^*]|\*(?!\/))*\*\//, 'comment'],

                // Strings: double-quoted
                [/"/, {token: 'string.quote', next: '@string_double'}],
                // Long strings: two single-quotes
                [/''/, {token: 'string.quote', next: '@string_long'}],

                // Interpolation start
                [/\$\{/, {token: 'delimiter.bracket', next: '@interpolation'}],

                // Numbers
                [/\b[0-9]+\b/, 'number'],

                // Paths and URLs
                [/~?[A-Za-z0-9_.\-+]+(\/[A-Za-z0-9_.\-+]+)+/, 'string.unquoted.path'],
                [/<[A-Za-z0-9_.\-+]+(\/[A-Za-z0-9_.\-+]+)*>/, 'string.unquoted.spath'],
                [/[A-Za-z][A-Za-z0-9+\-.]*:[A-Za-z0-9%\/\?:@&=+\$,\-_.!~\*']+/, 'string.unquoted.url'],

                // Identifiers and keywords
                [
                    /[a-zA-Z_][\w'\-]*/,
                    {
                        cases: {
                            '@keywords': 'keyword',
                            '@builtins': 'constant.language',
                            '@functions': 'support.function',
                            '@operators': 'operator',
                            '@default': 'identifier',
                        },
                    },
                ],

                // Operators
                [/[=><!~?:&|+\-*\/]+/, 'operator'],

                // Delimiters
                [/[;,.]/, 'delimiter'],

                // Brackets
                [/[{}()\[\]]/, '@brackets'],
            ],

            // Deal with whitespace
            whitespace: [[/[ \t\r\n]+/, 'white']],

            // Double-quoted string
            string_double: [[/\\./, 'string.escape'], [/"/, {token: 'string.quote', next: '@pop'}], {include: '@root'}],

            // Long string ('' ... '')
            string_long: [
                [/''/, {token: 'string.quote', next: '@pop'}],
                [/\\./, 'string.escape'],
                [/\$\{/, {token: 'delimiter.bracket', next: '@interpolation'}],
                {include: '@root'},
            ],

            // Interpolation inside strings
            interpolation: [[/\}/, {token: 'delimiter.bracket', next: '@pop'}], {include: 'root'}],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'nix'});
monaco.languages.setMonarchTokensProvider('nix', def);

export default def;
