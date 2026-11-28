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

// syntax highlighting for mirl's text form (`mach build --emit ir-opt` / `--emit ir`)
function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: '',

        keywords: [
            'mirl',
            'target',
            'language',
            'debug',
            'rec',
            'uni',
            'pub',
            'ext',
            'fun',
            'val',
            'var',
            'bind',
            'unavailable',
            'zero',
            'spec',
            'packed',
            'align',
            'at',
            'in',
            'holds',
            'parameter',
            'file',
            'loc',
            'site',
            'type',
            'own',
            'outside',
            'open',
            'discriminant',
            'payload',
            'field',
            'sext',
            'zext',
        ],

        opcodes: ['br', 'cbr', 'switch', 'ret', 'unreachable', 'call', 'alloca', 'load', 'store', 'select', 'asm'],

        typeKeywords: [
            'i1',
            'i8',
            'i16',
            'i32',
            'i64',
            'i128',
            'i256',
            'i512',
            'f16',
            'f32',
            'f64',
            'ptr',
            'vec',
            'handle',
        ],

        decorators: [
            'align',
            'ascending',
            'available',
            'bounded',
            'checked',
            'constant_time',
            'convention',
            'debug',
            'descending',
            'effects',
            'export',
            'extensions',
            'init',
            'inline',
            'kept',
            'merged',
            'placed',
            'public',
            'public_pointers',
            'replaceable',
            'result_follows',
            'scalar',
            'secret',
            'secret_parameters',
            'secret_result',
            'section',
            'tls',
            'unique',
            'visibility',
            'volatile',
            'weak',
            'within',
        ],

        tokenizer: {
            root: [
                [/#(?!\[).*$/, 'comment'],
                [/#\[/, {token: 'annotation', next: '@decorator'}],

                [/^\s*[a-zA-Z_]\w*(?=\s*(?:\([^)]*\))?:)/, 'type.identifier'],

                [/\?(?:[a-zA-Z_][\w.]*|"(?:[^"\\]|\\.)*")/, 'variable.predefined'],

                [/\b[a-z_]\w*(?:\.[a-z_]\w*)+/, 'keyword.operator'],

                [
                    /[a-zA-Z_]\w*/,
                    {
                        cases: {
                            '@keywords': 'keyword',
                            '@opcodes': 'keyword.operator',
                            '@typeKeywords': 'type',
                            '@default': 'identifier',
                        },
                    },
                ],

                [/"(?:[^"\\]|\\.)*"/, 'string'],
                [/-?0x[0-9a-fA-F_]+/, 'number.hex'],
                [/-?\d[\d_]*/, 'number'],

                [/::|\^/, 'operator'],
                [/[{}()[\]]/, '@brackets'],
                [/[,:;=.+-]/, 'delimiter'],
                [/\s+/, 'white'],
            ],

            decorator: [
                [/\]/, {token: 'annotation', next: '@pop'}],
                [/\[/, {token: 'annotation', next: '@decoratorBracket'}],
                [/[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+/, 'annotation'],
                [
                    /[a-zA-Z_]\w*/,
                    {cases: {'@decorators': 'annotation', '@typeKeywords': 'type', '@default': 'identifier'}},
                ],
                [/"(?:[^"\\]|\\.)*"/, 'string'],
                [/-?0x[0-9a-fA-F_]+/, 'number.hex'],
                [/-?\d[\d_]*/, 'number'],
                [/[(),:{}=]/, 'delimiter'],
                [/\s+/, 'white'],
            ],

            decoratorBracket: [
                [/\]/, {token: 'annotation', next: '@pop'}],
                [/\[/, {token: 'annotation', next: '@decoratorBracket'}],
                [/[^[\]]+/, 'annotation'],
            ],
        },
    };
}

monaco.languages.register({id: 'mach-ir'});
monaco.languages.setMonarchTokensProvider('mach-ir', definition());
