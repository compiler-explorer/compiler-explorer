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

// Monaco ships an `ini` language, but TOML's array-of-tables headers ([[bin]]), arrays and inline tables all come out
// wrong under it, and those are exactly what a Cargo.toml is full of.
function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: '',
        tokenPostfix: '.toml',

        brackets: [
            {open: '[', close: ']', token: 'delimiter.square'},
            {open: '{', close: '}', token: 'delimiter.curly'},
        ],

        escapes: /\\(?:[btnfr"\\]|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        tokenizer: {
            root: [
                [/#.*$/, 'comment'],

                // [table] and [[array.of.tables]]
                [/^\s*\[\[.*?\]\]/, 'metatag'],
                [/^\s*\[.*?\]/, 'metatag'],

                // key = , "quoted key" = , dotted.key =
                [/[A-Za-z0-9_-]+(?=\s*\.)/, 'key'],
                [/[A-Za-z0-9_-]+(?=\s*=)/, 'key'],
                [/"[^"]*"(?=\s*=)/, 'key'],
                [/'[^']*'(?=\s*=)/, 'key'],

                [/[=,.]/, 'delimiter'],
                [/[[\]{}]/, '@brackets'],

                {include: '@values'},

                [/[ \t\r\n]+/, ''],
            ],

            values: [
                [/(true|false)\b/, 'keyword'],

                // Offset date-time, local date-time, date, and time
                [/\d{4}-\d{2}-\d{2}([Tt ]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})?)?/, 'number.date'],
                [/\d{2}:\d{2}:\d{2}(\.\d+)?/, 'number.date'],

                [/0x[0-9A-Fa-f](_?[0-9A-Fa-f])*/, 'number.hex'],
                [/0o[0-7](_?[0-7])*/, 'number.octal'],
                [/0b[01](_?[01])*/, 'number.binary'],
                [/[+-]?(inf|nan)\b/, 'number.float'],
                [/[+-]?\d(_?\d)*\.\d(_?\d)*([eE][+-]?\d(_?\d)*)?/, 'number.float'],
                [/[+-]?\d(_?\d)*[eE][+-]?\d(_?\d)*/, 'number.float'],
                [/[+-]?\d(_?\d)*/, 'number'],

                [/"""/, 'string', '@multiLineBasic'],
                [/'''/, 'string', '@multiLineLiteral'],
                [/"/, 'string', '@basicString'],
                [/'/, 'string', '@literalString'],
            ],

            basicString: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop'],
            ],

            // Literal strings take no escapes at all.
            literalString: [
                [/[^']+/, 'string'],
                [/'/, 'string', '@pop'],
            ],

            multiLineBasic: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"""/, 'string', '@pop'],
                [/"/, 'string'],
            ],

            multiLineLiteral: [
                [/[^']+/, 'string'],
                [/'''/, 'string', '@pop'],
                [/'/, 'string'],
            ],
        },
    };
}

monaco.languages.register({id: 'toml'});
monaco.languages.setMonarchTokensProvider('toml', definition());
monaco.languages.setLanguageConfiguration('toml', {
    comments: {lineComment: '#'},
    brackets: [
        ['[', ']'],
        ['{', '}'],
    ],
    autoClosingPairs: [
        {open: '[', close: ']'},
        {open: '{', close: '}'},
        {open: '"', close: '"'},
        {open: "'", close: "'"},
    ],
});

export default definition();
