// Copyright (c) 2018, Mitch Kennedy
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

export function definition() {
    // Ada 2012 Language Definition
    return {
        keywords: [
            'abort',
            'else',
            'new',
            'return',
            'elsif',
            'reverse',
            'abstract',
            'end',
            'null',
            'accept',
            'entry',
            'select',
            'access',
            'exception',
            'of',
            'separate',
            'aliased',
            'exit',
            'some',
            'all',
            'others',
            'subtype',
            'for',
            'out',
            'synchronized',
            'array',
            'function',
            'overriding',
            'at',
            'tagged',
            'generic',
            'package',
            'task',
            'begin',
            'goto',
            'pragma',
            'terminate',
            'body',
            'private',
            'then',
            'if',
            'procedure',
            'type',
            'case',
            'in',
            'protected',
            'constant',
            'interface',
            'until',
            'is',
            'raise',
            'use',
            'declare',
            'range',
            'delay',
            'limited',
            'record',
            'when',
            'delta',
            'loop',
            'while',
            'digits',
            'renames',
            'with',
            'do',
            'requeue',
            'rem',
            'mod',
            'abs',
            'not',
            'and',
            'or',
            'xor',
        ],
        standardTypes: [
            // Defined in the package Standard
            // See: http://www.adaic.org/resources/add_content/standards/12rm/html/RM-A-1.html
            'Boolean',
            'Integer',
            'Natural',
            'Positive ',
            'Float',
            'Character',
            'Wide_Character',
            'Wide_Wide_Character',
            'String',
            'Wide_String',
            'Wide_Wide_String',
            'Duration',
            // Predefined Standard exceptions
            'Constraint_Error',
            'Program_Error',
            'Storage_Error',
            'Tasking_Error',
        ],

        operators: [
            '+',
            '-',
            '*',
            '/',
            'div',
            'mod',
            'shl',
            'shr',
            'and',
            'or',
            'xor',
            'not',
            '<',
            '>',
            '<=',
            '>=',
            '==',
            '<>',
            '+=',
            '-=',
            '*=',
            '/=',
        ],
        brackets: [
            ['(', ')', 'delimiter.parenthesis'],
            ['[', ']', 'delimiter.square'],
        ],
        symbols: /[=><!~&|+\-*/^]+/,
        delimiters: /[;=.:,`]/,
        escapes: /\\(?:[abfnrtv\\'\n\r]|x[0-9A-Fa-f]{2}|[0-7]{3}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|N\{\w+\})/,

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                [
                    /[a-zA-Z_][a-zA-Z0-9_]*/,
                    {
                        cases: {
                            '@standardTypes': 'type',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],
                // Whitespace
                {include: '@whitespace'},

                [/[()[\]]/, '@brackets'],

                // Numbers
                // See https://regex101.com/r/dflfeQ/2 for examples from the
                // 2012 ARM (http://www.ada-auth.org/standards/12rm/html/RM-2-4-1.html#S0009)
                [/[0-9_.]+(E[+-]?\d+)?/, 'number.float'],
                // See https://regex101.com/r/dSSADT/3 for examples from the
                // 2012 ARM (http://www.ada-auth.org/standards/12rm/html/RM-2-4-2.html#S0011)
                [/[0-9]+#[0-9A-Fa-f_.]+#(E[+-]?\d+)?/, 'number.hex'],

                [
                    /@delimiters/,
                    {
                        cases: {
                            '@keywords': 'keyword',
                            '@default': 'delimiter',
                        },
                    },
                ],
                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-terminated string
                [/"/, 'string', '@string'],

                // characters
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            // Whitespace and comments
            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/--.*$/, 'comment'],
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
monaco.languages.register({id: 'ada'});
monaco.languages.setMonarchTokensProvider('ada', definition());

export {};
