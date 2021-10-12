// Copyright (c) 2021, Compiler Explorer Authors
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
        tokenizer: {
            root: [
                [/^(\| )*==.*$/, 'comment'],
                [/^(\| )*catch type.*$/, 'comment'],
                [/^(\| )*local table.*$/, 'comment'],
                [/^(\| )*\[\s*\d+\].*$/, 'comment'],
                [/^(\| )*\|-+$/, 'comment'],
                [/^((?:\| )*)(\d+)/, ['comment', { token: 'number', next: '@opcode' }]],
                [/^((?:\| )*)(\d+)(\s+)/, ['comment', 'number', { token: '', next: '@opcode' }]],
            ],

            opcode: [
                [/[a-z_]\w*\s*$/, { token: 'keyword', next: '@root' }],
                [/([a-z_]\w*)(\s+)/, ['keyword', { token: '', next: '@arguments' }]],
            ],

            arguments: [
                [/(.*?)(\(\s*\d+\)(?:\[[^\]]+\])?)$/, ['', { token: 'comment', next: '@root' }]],
                [/.*$/, { token: '', next: '@root' }],
            ],
        },
    };
}

var def = definition();
monaco.languages.register({ id: 'asmruby' });
monaco.languages.setMonarchTokensProvider('asmruby', def);

module.exports = def;
