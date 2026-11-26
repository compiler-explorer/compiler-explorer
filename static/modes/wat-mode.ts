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

// Follows the lexical grammar of the WebAssembly text format:
// https://webassembly.github.io/spec/core/text/lexical.html
function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: '',

        brackets: [{open: '(', close: ')', token: 'delimiter.parenthesis'}],

        typeKeywords: [
            'i8',
            'i16',
            'i32',
            'i64',
            'f32',
            'f64',
            'v128',
            'funcref',
            'externref',
            'anyref',
            'eqref',
            'i31ref',
            'structref',
            'arrayref',
            'exnref',
            'nullref',
            'nullfuncref',
            'nullexternref',
            'nullexnref',
        ],

        idchar: /[0-9A-Za-z!#$%&'*+\-./:<=>?@\\^_`|~]/,
        num: /\d(?:_?\d)*/,
        hexnum: /[0-9a-fA-F](?:_?[0-9a-fA-F])*/,
        number: /[+-]?(?:0x@hexnum(?:\.(?:@hexnum)?)?(?:[pP][+-]?@num)?|@num(?:\.(?:@num)?)?(?:[eE][+-]?@num)?|inf|nan(?::0x@hexnum)?)/,
        escapes: /\\(?:[tnr"'\\]|u\{@hexnum\}|[0-9a-fA-F]{2})/,

        tokenizer: {
            root: [
                {include: '@whitespace'},

                // `@@` is Monarch's escape for a literal `@`
                [/(\()(@@@idchar+)/, ['@brackets', 'annotation']],
                [/[()]/, '@brackets'],

                [/"/, 'string', '@string'],

                [/@number(?!@idchar)/, 'number'],

                [/\$@idchar+/, 'identifier'],

                // Every bare word is a keyword: instructions, module fields and types alike.
                [/[a-z]@idchar*/, {cases: {'@typeKeywords': 'type', '@default': 'keyword'}}],

                // Anything else built from idchars is a reserved token, i.e. not valid WAT.
                [/@idchar+/, ''],
            ],

            whitespace: [
                [/\s+/, 'white'],
                [/;;.*$/, 'comment'],
                [/\(;/, 'comment', '@comment'],
            ],

            comment: [
                [/[^(;]+/, 'comment'],
                [/\(;/, 'comment', '@push'],
                [/;\)/, 'comment', '@pop'],
                [/[(;]/, 'comment'],
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'wat'});
monaco.languages.setMonarchTokensProvider('wat', def);

export default def;
