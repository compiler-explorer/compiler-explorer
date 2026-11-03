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

// Nexium: https://londopy.github.io/nexium/ (the token rules of self/lexer.nx)
function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: 'invalid',

        keywords: [
            'fn',
            'let',
            'var',
            'const',
            'struct',
            'enum',
            'record',
            'ref',
            'class',
            'trait',
            'impl',
            'pub',
            'import',
            'return',
            'if',
            'else',
            'for',
            'while',
            'match',
            'break',
            'continue',
            'try',
            'catch',
            'defer',
            'errdefer',
            'comptime',
            'unsafe',
            'error',
            'true',
            'false',
            'null',
            'undefined',
            'and',
            'or',
            'type',
            'distinct',
            'where',
            'into',
            'artifact',
            'test',
            'export',
            'unreachable',
            'as',
            'orelse',
            'in',
            'dyn',
            'weak',
            'parallel',
            'extern',
            'using',
            'own',
            'step',
            'derive',
            'layout',
        ],

        typeKeywords: [
            'i8',
            'i16',
            'i32',
            'i64',
            'i128',
            'u8',
            'u16',
            'u32',
            'u64',
            'u128',
            'isize',
            'usize',
            'f32',
            'f64',
            'bool',
            'char',
            'void',
            'never',
            'String',
            'List',
            'Map',
            'Self',
        ],

        operators: [
            '+',
            '-',
            '*',
            '/',
            '%',
            '+%',
            '-%',
            '*%',
            '+|',
            '-|',
            '*|',
            '&',
            '|',
            '^',
            '~',
            '<<',
            '>>',
            '=',
            '==',
            '!=',
            '<',
            '>',
            '<=',
            '>=',
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
            '+%=',
            '-%=',
            '*%=',
            '->',
            '=>',
            '|>',
            '..',
            '..=',
            '?',
            '!',
            '.?',
            '.*',
            '?.',
        ],

        symbols: /[=><!~?:&|+\-*/^%.]+/,

        escapes: /\\(?:[nrt0\\"']|x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]+\})/,

        tokenizer: {
            root: [
                // a declaration's name
                [/(fn)(\s+)([a-zA-Z_]\w*)/, ['keyword', 'white', 'entity.name.function']],
                [/(struct|enum|record|trait|error|class)(\s+)([A-Z]\w*)/, ['keyword', 'white', 'type.identifier']],
                // builtins: @cImport, @sizeOf; effect bounds: !allocates
                [/@[a-zA-Z_]\w*/, 'builtin.identifier'],
                [
                    /!(allocates|refcounts|blocks|shared_mutable|nondeterministic|panics|ffi|unbounded_stack)\b/,
                    'annotation',
                ],
                // identifiers and keywords
                [
                    /[a-z_]\w*/,
                    {
                        cases: {
                            '@typeKeywords': 'keyword.type',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],
                [
                    /[A-Z]\w*/,
                    {
                        cases: {
                            '@typeKeywords': 'keyword.type',
                            '@default': 'type.identifier',
                        },
                    },
                ],

                {include: '@whitespace'},

                // binary patterns and brackets
                [/<<|>>/, '@brackets'],
                [/[{}()[\]]/, '@brackets'],
                [/\.\{/, '@brackets'],
                [
                    /@symbols/,
                    {
                        cases: {
                            '@operators': 'operator',
                            '@default': '',
                        },
                    },
                ],

                // numbers
                [/\d[\d_]*\.\d[\d_]*([eE][-+]?\d+)?/, 'number.float'],
                [/\d[\d_]*[eE][-+]?\d+/, 'number.float'],
                [/0[xX][0-9a-fA-F_]+/, 'number.hex'],
                [/0[oO][0-7_]+/, 'number.octal'],
                [/0[bB][01_]+/, 'number.binary'],
                [/\d[\d_]*/, 'number'],

                [/[;,]/, 'delimiter'],

                // strings: "…" with {} placeholders, b"…" bytes, r"…" raw
                [/r"[^"]*"/, 'string'],
                [/b?"([^"\\]|\\.)*$/, 'string.invalid'],
                [/b?"/, 'string', '@string'],

                // chars
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\/\/[!/].*$/, 'comment.doc'],
                [/\/\/.*$/, 'comment'],
            ],

            string: [
                [/[^\\"{}]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/\{\{|\}\}/, 'string.escape'],
                [/\{[^}"]*\}/, 'variable'],
                [/[{}]/, 'string'],
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'nexium'});
monaco.languages.setMonarchTokensProvider('nexium', def);

export default def;
