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
'use strict';
var monaco = require('monaco-editor');

function definition() {
    return {
        defaultToken: 'invalid',

        keywords: [
            'abstract',
            'alias',
            'align',
            'asm',
            'assert',
            'auto',
            'body',
            'bool',
            'break',
            'byte',
            'case',
            'cast',
            'catch',
            'cdouble',
            'cent',
            'cfloat',
            'char',
            'class',
            'const',
            'continue',
            'creal',
            'dchar',
            'debug',
            'default',
            'delegate',
            'delete ',
            'deprecated',
            'do',
            'double',
            'else',
            'enum',
            'export',
            'extern',
            'false',
            'final',
            'finally',
            'float',
            'for',
            'foreach',
            'foreach_reverse',
            'function',
            'goto',
            'idouble',
            'if',
            'ifloat',
            'immutable',
            'import',
            'in',
            'inout',
            'int',
            'interface',
            'invariant',
            'ireal',
            'is',
            'lazy',
            'long',
            'macro',
            'mixin',
            'module',
            'new',
            'nothrow',
            'null',
            'out',
            'override',
            'package',
            'pragma',
            'private',
            'protected',
            'public',
            'pure',
            'real',
            'ref',
            'return',
            'scope',
            'shared',
            'short',
            'static',
            'struct',
            'super',
            'switch',
            'synchronized',
            'template',
            'this',
            'throw',
            'true',
            'try',
            'typedef',
            'typeid',
            'typeof',
            'ubyte',
            'ucent',
            'uint',
            'ulong',
            'union',
            'unittest',
            'ushort',
            'version',
            'void',
            'volatile',
            'wchar',
            'while',
            'with',
            '__FILE__',
            '__FILE_FULL_PATH__',
            '__MODULE__',
            '__LINE__',
            '__FUNCTION__',
            '__PRETTY_FUNCTION__',
            '__gshared',
            '__traits',
            '__vector',
            '__parameters'],

        typeKeywords: [
            'bool', 'byte', 'ubyte', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'char', 'wchar', 'dchar',
            'float', 'double', 'real', 'ifloat', 'idouble', 'ireal', 'cfloat', 'cdouble', 'creal', 'void'
        ],

        operators: [
            '=', '>', '<', '!', '~', '?', ':',
            '==', '<=', '>=', '!=', '&&', '||', '++', '--',
            '+', '-', '*', '/', '&', '|', '^', '%', '<<',
            '>>', '>>>', '+=', '-=', '*=', '/=', '&=', '|=',
            '^=', '%=', '<<=', '>>=', '>>>='
        ],

        // we include these common regular expressions
        symbols: /[!%&*+/:<=>?^|~-]+/,
        escapes: /\\(?:["'\\abfnrtv]|x[\dA-Fa-f]{1,4}|u[\dA-Fa-f]{4}|U[\dA-Fa-f]{8})/,

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                // identifiers and keywords
                [/[$_a-z][\w$]*/, {
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
                [/[()[\]{}]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],
                [/@symbols/, {
                    cases: {
                        '@operators': 'operator',
                        '@default': ''
                    }
                }],

                // numbers
                [/\d*\.\d+([Ee][+-]?\d+)?[DFdf]?/, 'number.float'],
                [/0[Xx][\dA-F_a-f]*[\dA-Fa-f][Ll]?/, 'number.hex'],
                [/0[0-7_]*[0-7][Ll]?/, 'number.octal'],
                [/0[Bb][01_]*[01][Ll]?/, 'number.binary'],
                [/\d+[Ll]?/, 'number'],

                // delimiter: after number because of .\d floats
                [/[,.;]/, 'delimiter'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'],  // non-teminated string
                [/"/, 'string', '@string'],
                [/`/, 'string', '@rawstring'],

                // characters
                [/'[^'\\]'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid']
            ],

            whitespace: [
                [/[\t\n\r ]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\+/, 'comment', '@nestingcomment'],
                [/\/\/.*$/, 'comment']
            ],

            comment: [
                [/[^*/]+/, 'comment'],
                [/\*\//, 'comment', '@pop'],
                [/[*/]/, 'comment']
            ],

            nestingcomment: [
                [/[^+/]+/, 'comment'],
                [/\/\+/, 'comment', '@push'],
                [/\/\+/, 'comment.invalid'],
                [/\+\//, 'comment', '@pop'],
                [/[+/]/, 'comment']
            ],

            string: [
                [/[^"\\]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop']
            ],

            rawstring: [
                [/[^`]/, 'string'],
                [/`/, 'string', '@pop']
            ]
        }
    };
}

function configuration() {
    return {
        comments: {
            lineComment: '//',
            blockComment: ['/*', '*/']
        },

        brackets: [
            ['{', '}'],
            ['[', ']'],
            ['(', ')']
        ],

        autoClosingPairs: [
            {open: '{', close: '}'},
            {open: '[', close: ']'},
            {open: '(', close: ')'},
            {open: '`', close: '`', notIn: ['string']},
            {open: '"', close: '"', notIn: ['string']},
            {open: '\'', close: '\'', notIn: ['string', 'comment']}
        ],

        surroundingPairs: [
            {open: '{', close: '}'},
            {open: '[', close: ']'},
            {open: '(', close: ')'},
            {open: '`', close: '`'},
            {open: '"', close: '"'},
            {open: '\'', close: '\''}
        ]
    };
}

monaco.languages.register({id: 'd'});
monaco.languages.setMonarchTokensProvider('d', definition());
monaco.languages.setLanguageConfiguration('d', configuration());
