// Copyright (c) 2026, Compiler Explorer Authors
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

// Salam source may be written with either English or Persian keywords, so both
// spellings are recognized here (see SalamLang/Salam compiler/langpack.salam).
function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: '',

        keywords: [
            // English keywords
            'func',
            'ret',
            'if',
            'else',
            'until',
            'on',
            'mut',
            'const',
            'type',
            'struct',
            'enum',
            'end',
            'import',
            'as',
            'true',
            'false',
            'null',
            'this',
            'break',
            'continue',
            'layout',
            'package',
            'print',
            'println',
            'printerr',
            'printerrln',
            'input',
            'defer',
            'operator',
            'extern',
            'interface',
            'pub',
            'inline',
            'noinline',
            'pure',
            'noret',
            'deprecated',
            'component',
            'repeat',
            'impl',
            'to',
            'by',
            'each',
            'in',
            'with',
            'match',
            'while',
            'link',
            'static',
            'dynamic',
            'framework',
            'kind',
            // Persian keywords
            'کارکرد',
            'بازگشت',
            'اگر',
            'وگرنه',
            'تاهنگام',
            'بر',
            'گذرا',
            'پایدار',
            'ریخت',
            'ساختار',
            'شمارش',
            'پایان',
            'فراخوانی',
            'برگردان',
            'درست',
            'نادرست',
            'پوچ',
            'این',
            'بشکن',
            'بگذر',
            'چیدمان',
            'بسته',
            'بنویس',
            'چاپ',
            'نادرستینویس',
            'نادرستیچاپ',
            'بخوان',
            'دیرکرد',
            'کنشگر',
            'بیرونی',
            'میانجی',
            'همگانی',
            'توکار',
            'جدا',
            'درونزا',
            'بی‌بازگشت',
            'ازکارافتاده',
            'سازه',
            'چرخه',
            'پیاده‌سازی',
            'تا',
            'گام',
            'هر',
            'در',
            'با',
            'برگزین',
            'و',
            'یا',
            'پیوند',
            'ایستا',
            'پویا',
            'چارچوب',
            'گونه',
        ],

        typeKeywords: [
            'int',
            'str',
            'bool',
            'float',
            'byte',
            'i8',
            'i16',
            'i32',
            'i64',
            'u8',
            'u16',
            'u32',
            'u64',
            'f32',
            'f64',
        ],

        operators: [
            '=',
            ':=',
            '+',
            '-',
            '*',
            '/',
            '%',
            '==',
            '!=',
            '<',
            '<=',
            '>',
            '>=',
            '!',
            '&',
            '&&',
            '|',
            '||',
            '+=',
            '-=',
            '*=',
            '/=',
            '.',
            '..',
            '&:',
        ],

        symbols: /[=><!~?:&|+\-*/^%]+/,
        // ASCII, Arabic-Indic and Extended Arabic-Indic (Persian) digits, e.g. "1_000", "۱_۰۰۰".
        digits: /[0-9٠-٩۰-۹]/,
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        tokenizer: {
            root: [
                // Latin identifiers and keywords
                [
                    /[a-zA-Z_$][\w$]*/,
                    {cases: {'@typeKeywords': 'keyword.type', '@keywords': 'keyword', '@default': 'identifier'}},
                ],
                // Persian/Arabic identifiers and keywords (letters + ZWNJ for compounds like "بی‌بازگشت")
                [/[؀-ۿ][؀-ۿ‌]*/, {cases: {'@keywords': 'keyword', '@default': 'identifier'}}],

                {include: '@whitespace'},

                [/[{}()[\]]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],
                [/@symbols/, {cases: {'@operators': 'operator', '@default': ''}}],

                [/@digits+\.@digits+([eE][-+]?@digits+)?/, 'number.float'],
                [/0[xX][0-9a-fA-F_]+/, 'number.hex'],
                [/(@digits|_)+/, 'number'],

                [/[;,]/, 'delimiter'],

                [/"([^"\\]|\\.)*$/, 'string.invalid'],
                [/"/, {token: 'string.quote', bracket: '@open', next: '@string'}],

                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            comment: [
                [/[^/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'],
                [/\*\//, 'comment', '@pop'],
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

monaco.languages.register({id: 'salam'});
monaco.languages.setMonarchTokensProvider('salam', definition());
monaco.languages.setLanguageConfiguration('salam', configuration());
