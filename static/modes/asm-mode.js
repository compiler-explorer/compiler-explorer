// Copyright (c) 2012, Matt Godbolt
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
        // Set defaultToken to invalid to see what you do not tokenize yet
        defaultToken: 'invalid',

        // C# style strings
        escapes: /\\(?:["'\\abfnrtv]|x[\dA-Fa-f]{1,4}|u[\dA-Fa-f]{4}|U[\dA-Fa-f]{8})/,

        registers: /%?\b(r\d+[bdw]?|([er]?([a-d][hlx]|cs|fs|ds|ss|sp|bp|ip|sil?|dil?))|[x-z]mm\d+|sp|fp|lr)\b/,

        intelOperators: /PTR|(D|Q|[X-Z]MM)?WORD/,

        tokenizer: {
            root: [
                // Error document
                [/^<.*>$/, {token: 'annotation'}],
                // Label definition
                [/^[\w$.?@].*:/, {token: 'type.identifier'}],
                // Label definition (ARM style)
                [/^\s*\|[^|]*\|/, {token: 'type.identifier'}],
                // Label definition (CL style)
                [/^\s*[\w$.|]*\s*(PROC|ENDP|DB|DD)/, {token: 'type.identifier'}],
                // Constant definition
                [/^[\w$.?@][^=]*=/, {token: 'type.identifier'}],
                // opcode
                [/[.A-Z_a-z][\w.]*/, {token: 'keyword', next: '@rest'}],
                // braces and parentheses at the start of the line (e.g. nvcc output)
                [/[(){}]/, {token: 'operator', next: '@rest'}],

                // whitespace
                {include: '@whitespace'}
            ],

            rest: [
                // pop at the beginning of the next line and rematch
                [/^.*$/, {token: '@rematch', next: '@pop'}],

                [/@registers/, 'variable.predefined'],
                [/@intelOperators/, 'annotation'],
                // brackets
                [/[()<>[\]{}]/, '@brackets'],

                // ARM-style label reference
                [/\|[^|]*\|*/, 'type.identifier'],

                // numbers
                [/\d*\.\d+([Ee][+-]?\d+)?/, 'number.float'],
                [/(\$|0[Xx])[\dA-Fa-f]+/, 'number.hex'],
                [/\d+/, 'number'],
                // ARM-style immediate numbers (which otherwise look like comments)
                [/#-?\d+/, 'number'],

                // operators
                [/[!&()*+,/:{}-]/, 'operator'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'],  // non-terminated string
                [/"/, {token: 'string.quote', bracket: '@open', next: '@string'}],

                // characters
                [/'[^'\\]'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],

                // Assume anything else is a label reference
                [/%?[$.?@A-Z_a-z][\w$.?@]*/, 'type.identifier'],

                // whitespace
                {include: '@whitespace'}
            ],

            comment: [
                [/[^*/]+/, 'comment'],
                [/\/\*/, 'comment', '@push'],    // nested comment
                ['\\*/', 'comment', '@pop'],
                [/[*/]/, 'comment']
            ],

            string: [
                [/[^"\\]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, {token: 'string.quote', bracket: '@close', next: '@pop'}]
            ],

            whitespace: [
                [/[\t\n\r ]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment'],
                [/[#;@\\].*$/, 'comment']
            ]
        }
    };
}

var def = definition();
monaco.languages.register({id: 'asm'});
monaco.languages.setMonarchTokensProvider('asm', def);

module.exports = def;
