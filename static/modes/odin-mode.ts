// Copyright (c) 2024, Compiler Explorer Authors
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
    // Odin language definition
    return {
        defaultToken: 'invalid',

        keywords: [
            'import',
            'export',
            'foreign',
            'package',
            'if',
            'else',
            'when',
            'for',
            'in',
            'defer',
            'switch',
            'return',
            'const',
            'fallthrough',
            'break',
            'continue',
            'case',
            'vector',
            'static',
            'dynamic',
            'using',
            'do',
            'inline',
            'no_inline',
            'asm',
            'yield',
            'await',
            'distinct',
            'context',
            'nil',
            'true',
            'false',
            'type',
            'var',
            'macro',
            'struct',
            'enum',
            'union',
            'set',
            'typeid',
            'cast',
            'transmute',
            'auto_cast',
            'proc',
        ],

        typeKeywords: [
            'bool',
            'byte',
            'b8',
            'b16',
            'b32',
            'b64',
            'int',
            'i8',
            'i16',
            'i32',
            'i64',
            'i128',
            'uint',
            'u8',
            'u16',
            'u32',
            'u64',
            'u128',
            'uintptr',
            'i16le',
            'i32le',
            'i64le',
            'i128le',
            'u16le',
            'u32le',
            'u64le',
            'u128le',
            'i16be',
            'i32be',
            'i64be',
            'i128be',
            'u16be',
            'u32be',
            'u64be',
            'u128be',
            'f16',
            'f32',
            'f64',
            'f16le',
            'f32le',
            'f64le',
            'f16be',
            'f32be',
            'f64be',
            'complex32',
            'complex64',
            'complex128',
            'quaternion64',
            'quaternion128',
            'quaternion256',
            'rune',
            'string',
            'cstring',
            'rawptr',
            'any',
            'bit_set',
            'bit_field',
            'map',
        ],

        operators: [
            '@',
            '|',
            '!',
            ':',
            '+',
            '-',
            '->',
            '*',
            '/',
            '<',
            '<<',
            '>',
            '>>',
            '~',
            '|=',
            '!=',
            ':=',
            '+=',
            '-=',
            '->=',
            '*=',
            '/=',
            '<=',
            '<<=',
            '>=',
            '>>=',
            '~=',
            '=',
            ':',
            '::',
            '..',
        ],

        // we include these common regular expressions
        symbols: /[=><!~?:&|+\-*/^%]+/,
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                // identifiers and keywords
                [
                    /[a-z_$][\w$]*/,
                    {cases: {'@typeKeywords': 'keyword', '@keywords': 'keyword', '@default': 'identifier'}},
                ],
                [/---$/, {cases: {'@typeKeywords': 'keyword', '@keywords': 'keyword', '@default': 'identifier'}}],

                // whitespace
                {include: '@whitespace'},

                // delimiters and operators
                [/[{}()[\]]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],

                [/@symbols/, {cases: {'@operators': 'type', '@default': ''}}],

                // # annotations.
                [/#\s*[a-zA-Z_$][\w$]*/, {token: 'annotation', log: 'annotation token: $0'}],
                [/@\(\s*[a-zA-Z_=$][\w=$]*\)/, {token: 'annotation', log: 'annotation token: $0'}],

                // numbers
                [/\d*\.\d+([eE][-+]?\d+)?/, 'number.float'],
                [/0[xX][0-9a-fA-F]+/, 'number.hex'],
                [/\d+/, 'number'],

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                [/"/, {token: 'string.quote', bracket: '@open', next: '@string'}],

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

function configuration(): monaco.languages.LanguageConfiguration {
    return {
        comments: {
            lineComment: '//',
            blockComment: ['/*', '*/'],
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

monaco.languages.register({id: 'odin'});
monaco.languages.setMonarchTokensProvider('odin', definition());
monaco.languages.setLanguageConfiguration('odin', configuration());

export {};
