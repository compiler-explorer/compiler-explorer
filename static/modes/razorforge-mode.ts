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

function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: '',

        keywords: [
            // Declarations
            'routine',
            'entity',
            'record',
            'choice',
            'flags',
            'crashable',
            'variant',
            'protocol',
            // Bindings
            'var',
            'preset',
            'lateinit',
            // Visibility / placement
            'secret',
            'posted',
            'common',
            // Self references
            'me',
            'Me',
            // Conformance and constraints
            'obeys',
            'disobeys',
            'needs',
            'relates',
            // Control flow
            'if',
            'elseif',
            'else',
            'then',
            'unless',
            'when',
            'is',
            'isnot',
            'isonly',
            'loop',
            'while',
            'for',
            'break',
            'continue',
            'return',
            'throw',
            'absent',
            'becomes',
            // Modules
            'import',
            'module',
            // Misc
            'using',
            'as',
            'define',
            'pass',
            'with',
            'given',
            'in',
            'notin',
            'to',
            'til',
            'by',
            'discard',
            // Logical operators
            'and',
            'or',
            'not',
            'but',
            // Literals
            'true',
            'false',
            'None',
            'none',
            // Memory / interop
            'dangerous',
            'external',
            'steal',
        ],

        operators: [
            // Comparison
            '==',
            '!=',
            '<',
            '<=',
            '>',
            '>=',
            '<=>',
            // Arithmetic (checked by default)
            '+',
            '-',
            '*',
            '/',
            '//',
            '%',
            '**',
            // Wrapping variants
            '+%',
            '-%',
            '*%',
            '**%',
            // Clamping variants
            '+^',
            '-^',
            '*^',
            '**^',
            // Unchecked variants
            '+!',
            '-!',
            '*!',
            '/!',
            '//!',
            '%!',
            '**!',
            // Bitwise and shifts
            '&',
            '|',
            '^',
            '~',
            '<<',
            '>>',
            '<<<',
            '>>>',
            // Assignment
            '=',
            '+=',
            '-=',
            '*=',
            '/=',
            '//=',
            '%=',
            '**=',
            '+%=',
            '-%=',
            '*%=',
            '**%=',
            '+^=',
            '-^=',
            '*^=',
            '/^=',
            '**^=',
            '&=',
            '|=',
            '^=',
            '<<=',
            '>>=',
            '<<<=',
            '>>>=',
            '??=',
            // Failure and optionals
            '!',
            '!!',
            '?',
            '??',
            // Arrows
            '->',
            '=>',
        ],

        symbols: /[=><!~?&|+\-*/^%]+/,
        escapes: /\\(?:[abfnrtv\\"'0]|x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]+\})/,

        tokenizer: {
            root: [
                // Annotations: @inline, @readonly, @[readonly, innate]
                [/@\[[^\]]*\]/, 'annotation'],
                [/@[a-zA-Z_]\w*/, 'annotation'],

                // Compiler-special members like $create, $destroy, $eq
                [/\$[a-z_]\w*/, 'identifier.special'],

                // Capitalized keywords first, then types (S64, Text, user types)
                [/Me\b/, 'keyword'],
                [/None\b/, 'keyword'],
                [/[A-Z]\w*/, 'type.identifier'],

                // Identifiers and keywords
                [
                    /[a-z_]\w*/,
                    {
                        cases: {
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],

                // Whitespace and comments
                {include: '@whitespace'},

                // Delimiters and brackets
                [/[{}()[\]]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],
                [
                    /@symbols/,
                    {
                        cases: {
                            '@operators': 'operator',
                            '@default': '',
                        },
                    },
                ],

                // Numbers — base-prefixed, floats, then integers; literal suffixes
                // (1_s64, 2.5f32, 100ms, 4gib, 3j64, 1dn ...) ride along with \w*
                [/0[xX][0-9a-fA-F_]+\w*/, 'number.hex'],
                [/0[oO][0-7_]+\w*/, 'number.octal'],
                [/0[bB][01_]+\w*/, 'number.binary'],
                [/\d[\d_]*\.\d[\d_]*([eE][-+]?\d+)?\w*/, 'number.float'],
                [/\d[\d_]*\w*/, 'number'],

                [/[;,.:]/, 'delimiter'],

                // Strings: plain plus r/f/rf/b/br prefixes
                [/(?:rf|br|[rfb])?"([^"\\]|\\.)*$/, 'string.invalid'],
                [/(?:rf|br|[rfb])?"/, 'string', '@string'],

                // Character literals
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            whitespace: [
                [/[ \r\n]+/, 'white'],
                [/###.*$/, 'comment.doc'],
                [/#.*$/, 'comment'],
            ],

            string: [
                [/[^\\"{}]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/\{[^}]*\}/, 'variable'],
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

monaco.languages.register({id: 'razorforge'});
monaco.languages.setMonarchTokensProvider('razorforge', definition());
