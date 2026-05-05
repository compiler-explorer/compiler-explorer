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
//
// Generated from sleepy4727/supper-algol-68 (algol68.tmLanguage.json) by
// etc/scripts/tm-to-monarch.js — best-effort conversion, hand-edits welcome.

import * as monaco from 'monaco-editor';

export function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: '',
        tokenPostfix: '.algol68',
        tokenizer: {
            root: [
                {include: '@comments'},
                {include: '@strings'},
                {include: '@numbers'},
                {include: '@bits'},
                {include: '@keywords'},
                {include: '@types'},
                {include: '@operators'},
                {include: '@identifiers'},
            ],
            block_comment_content_1: [{include: '@block_comment_content'}, [/}/, {token: '', next: '@pop'}]],
            block_comment_content: [[/{/, {token: '', next: '@block_comment_content_1'}]],
            comments_2: [{include: '@block_comment_content'}, [/}/, {token: 'comment', next: '@pop'}]],
            comments: [[/{/, {token: 'comment', next: '@comments_2'}]],
            strings_3: [
                [/'.|'\((u[0-9a-f]{4}|U[0-9a-f]{8})(,(u[0-9a-f]{4}|U[0-9a-f]{8})+)*\)/, 'string.escape'],
                [/"/, {token: 'string', next: '@pop'}],
            ],
            strings: [[/"/, {token: 'string', next: '@strings_3'}]],
            numbers: [
                [/\b\d+\.\d*(?:[Ee][+-]?\d+)?\b|\b\d+(?:[Ee][+-]?\d+)\b/, 'number.float'],
                [/\b\d+\b/, 'number'],
            ],
            bits: [
                [/\b(?:16r)[ 0-9a-f]+\b/, 'number.hex'],
                [/\b(?:8r)[ 0-7]+\b/, 'number.octal'],
                [/\b(?:4r)[ 0-3]+\b/, 'number'],
                [/\b(?:2r)[ 0-1]+\b/, 'number.binary'],
            ],
            keywords: [
                [/\b(?:case|do|elif|else|for|from|go\s*to|if|while)\b/, 'keyword'],
                [
                    /\b(?:access|at|begin|by|co|comment|def|egg|empty|end|esac|exit|fed|fi|flex|format|in|mode|module|nest|od|of|op|ouse|out|par|pr|pragmat|prio|proc|skip|then|to)\b/,
                    'keyword',
                ],
                [/\b(?:false|nil|true)\b/, 'keyword'],
            ],
            types: [
                [/\b(?:bits|bool|bytes|channel|char|compl|file|int|real|sema|string|struct|union|void)\b/, 'type'],
                [/\b(?:flex|heap|loc|long|ref|short)\b/, 'keyword'],
            ],
            operators: [
                [/\b(?::=)\b/, 'operator'],
                [/\b(?:<=|>=|\/=|=|<|>|:=:|:\/=:)\b/, 'operator'],
                [/\b(?:\+|-|\*|\/|OVER|MOD)\b/, 'operator'],
                [/\b(?:AND|OR|NOT|&)\b/, 'operator'],
            ],
            identifiers: [
                [/\b(?:[A-Z]_?(?:[A-Za-z0-9]_?)*)\b/, 'identifier'],
                [/\b(?:[a-z]_?(?:[a-z0-9]_?)*)\b/, 'identifier'],
            ],
        },
    };
}
monaco.languages.register({id: 'algol68'});
monaco.languages.setMonarchTokensProvider('algol68', definition());
