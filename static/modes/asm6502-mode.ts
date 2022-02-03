// Copyright (c) 2019, Compiler Explorer Authors
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
        tokenizer: {
            root: [
                [/^[a-zA-Z_][a-zA-Z_0-9]*/, { token: 'type.identifier' }],
                [/^\s+/, { token: 'whitespace', next: '@opcode' }],
                [/^\.[a-zA-Z]+\s*/, { token: 'type.identifier', next: '@arguments' }],
                [/;.*$/, { token: 'comment', next: '@root' }],
                [/\s*,\s*/, { token: 'delimiter', next: '@arguments' }],
            ],

            opcode: [
                [/[a-zA-Z]+$/, { token: 'keyword', next: '@root' }],
                [/[a-zA-Z]+\s*/, { token: 'keyword', next: '@arguments' }],
                [/;.*$/, { token: 'comment', next: '@root' }],
            ],

            arguments: [
                [/\$#[0-9a-fA-F]+/, { token: 'number', next: '@root' }],
                [/\$[0-9a-fA-F]+/, { token: 'number', next: '@root' }],
                [/#\$[0-9a-fA-F]+/, { token: 'number', next: '@root' }],
                [/#<[0-9]+\(%[a-zA-Z]+\)/, { token: 'number', next: '@root' }],
                [/#>[0-9]+\(%[a-zA-Z]+\)/, { token: 'number', next: '@root' }],
                [/#<\$[0-9a-fA-F]+/, { token: 'number', next: '@root' }],
                [/#>\$[0-9a-fA-F]+/, { token: 'number', next: '@root' }],
                [/#%[0-1]+/, { token: 'number', next: '@root' }],
                [/#[0-9]+/, { token: 'number', next: '@root' }],
                [/[a-zA-Z_][a-zA-Z_0-9]*/, { token: 'type.identifier', next: '@root' }],
                [/[0-9]+/, { token: 'number', next: '@root' }],
                [/;.*$/, { token: 'comment', next: '@root' }],
            ],
        },
    };
}

const def = definition();
monaco.languages.register({ id: 'asm6502' });
monaco.languages.setMonarchTokensProvider('asm6502', def);

export = def;
