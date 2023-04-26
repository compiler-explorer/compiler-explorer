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
        defaultToken: '',

        brackets: [
            {open: '{', close: '}', token: 'delimiter.curly'},
            {open: '[', close: ']', token: 'delimiter.square'},
            {open: '(', close: ')', token: 'delimiter.parenthesis'},
            {open: '<', close: '>', token: 'delimiter.angle'},
        ],

        keywords: [
            'class',
            'delegate',
            'enum',
            'errordomain',
            'interface',
            'namespace',
            'signal',
            'struct',
            'using',
            'null',
            // modifiers
            'abstract',
            'async',
            'const',
            'dynamic',
            'extern',
            'inline',
            'internal',
            'out',
            'override',
            'owned',
            'private',
            'protected',
            'public',
            'ref',
            'sealed',
            'static',
            'unowned',
            'virtual',
            'volatile',
            'weak',
            // others
            'as',
            'base',
            'break',
            'case',
            'catch',
            'construct',
            'continue',
            'default',
            'delete',
            'do',
            'else',
            'ensures',
            'finally',
            'for',
            'foreach',
            'get',
            'if',
            'in',
            'is',
            'lock',
            'new',
            'params',
            'requires',
            'return',
            'set',
            'sizeof',
            'switch',
            'this',
            'throw',
            'throws',
            'try',
            'typeof',
            'unlock',
            'value',
            'var',
            'while',
            'with',
            'yield',
            // primitives
            'bool',
            'char',
            'double',
            'float',
            'int',
            'int8',
            'int16',
            'int32',
            'int64',
            'long',
            'short',
            'size_t',
            'ssize_t',
            'string',
            'string16',
            'string32',
            'time_t',
            'uchar',
            'uint',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
            'ulong',
            'unichar',
            'unichar2',
            'ushort',
            'va_list',
            'void',
        ],

        namespaceFollows: ['namespace', 'using'],

        parenFollows: ['if', 'for', 'while', 'switch', 'foreach', 'using', 'catch', 'requires', 'ensures'],

        operators: [
            '=',
            '??',
            '||',
            '&&',
            '|',
            '^',
            '&',
            '==',
            '!=',
            '<=',
            '>=',
            '<<',
            '+',
            '-',
            '*',
            '/',
            '%',
            '!',
            '~',
            '++',
            '--',
            '+=',
            '-=',
            '*=',
            '/=',
            '%=',
            '&=',
            '|=',
            '^=',
            '<<=',
            '>>=',
            '>>',
            '=>',
        ],

        symbols: /[=><!~?:&|+\-*/^%]+/,

        // escape sequences
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                // identifiers and keywords
                [
                    /@?[a-zA-Z_]\w*/,
                    {
                        cases: {
                            '@namespaceFollows': {token: 'keyword.$0', next: '@namespace'},
                            '@keywords': {token: 'keyword.$0', next: '@qualified'},
                            '@default': {token: 'identifier', next: '@qualified'},
                        },
                    },
                ],

                // whitespace
                {include: '@whitespace'},

                // delimiters and operators
                [/[{}()[\]]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],
                [
                    /@symbols/,
                    {
                        cases: {
                            '@operators': 'delimiter',
                            '@default': '',
                        },
                    },
                ],

                // numbers
                [/[0-9_]*\.[0-9_]+([eE][-+]?\d+)?[fFdD]?/, 'number.float'],
                [/0[xX][0-9a-fA-F_]+/, 'number.hex'],
                [/0[bB][01_]+/, 'number.hex'], // binary: use same theme style as hex
                [/[0-9_U]+/, 'number'],

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                [/"""/, {token: 'string.quote', next: '@litstring'}],
                [/@"/, {token: 'string.quote', next: '@interpolatedstring'}],
                [/"/, {token: 'string.quote', next: '@string'}],

                // characters
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            qualified: [
                [
                    /[a-zA-Z_][\w]*/,
                    {
                        cases: {
                            '@keywords': {token: 'keyword.$0'},
                            '@default': 'identifier',
                        },
                    },
                ],
                [/\./, 'delimiter'],
                ['', '', '@pop'],
            ],

            namespace: [{include: '@whitespace'}, [/[A-Z]\w*/, 'namespace'], [/[.=]/, 'delimiter'], ['', '', '@pop']],

            comment: [
                [/[^/*]+/, 'comment'],
                ['\\*/', 'comment', '@pop'],
                [/[/*]/, 'comment'],
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, {token: 'string.quote', next: '@pop'}],
            ],

            litstring: [
                [/[^"]+/, 'string'],
                [/"""/, {token: 'string.quote', next: '@pop'}],
            ],

            interpolatedstring: [
                [/[^\\"$]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/\$/, 'string.escape'],
                [/"/, {token: 'string.quote', next: '@pop'}],
            ],

            whitespace: [
                [/[ \t\v\f\r\n]+/, ''],
                [/\/\*/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment'],
            ],
        },
    };
}

monaco.languages.register({id: 'vala'});
monaco.languages.setMonarchTokensProvider('vala', definition());

export {};
