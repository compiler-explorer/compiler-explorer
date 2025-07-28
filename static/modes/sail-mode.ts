// Copyright (c) 2025, Compiler Explorer Authors
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
        keywords: [
            'and',
            'as',
            'assert',
            'backwards',
            'barr',
            'bitfield',
            'bitone',
            'bitzero',
            'Bool',
            'by',
            'cast',
            'catch',
            'clause',
            'config',
            'configuration',
            'constant',
            'constraint',
            'dec',
            'default',
            'depend',
            'do',
            'eamem',
            'effect',
            'else',
            'end',
            'enum',
            'escape',
            'exmem',
            'forall',
            'foreach',
            'forwards',
            'from',
            'function',
            'if',
            'implicit',
            'impure',
            'in',
            'inc',
            'infix',
            'infixl',
            'infixr',
            'Int',
            'let',
            'mapping',
            'match',
            'newtype',
            'nondet',
            'operator',
            'Order',
            'overload',
            'pure',
            'ref',
            'register',
            'repeat',
            'return',
            'rmem',
            'rmemt',
            'rreg',
            'scattered',
            'sizeof',
            'struct',
            'then',
            'throw',
            'to',
            'try',
            'type',
            'Type',
            'undef',
            'undefined',
            'union',
            'unspec',
            'until',
            'val',
            'var',
            'where',
            'while',
            'with',
            'wmem',
            'wmv',
            'wmvt',
            'wreg',
        ],

        types: [
            'vector',
            'bitvector',
            'int',
            'nat',
            'atom',
            'range',
            'unit',
            'bit',
            'real',
            'list',
            'bool',
            'string',
            'bits',
            'option',
        ],

        identifier: "[a-zA-Z?_][a-zA-Z?0-9_'#]*",

        // String escape sequences. \\, \", \', \n, \t, \b, \r, \<newline>
        // Or hex/dec: \xHHH, \DD
        string_escapes: /\\(?:[\\"'ntbr]|$|x[0-9A-Fa-f]{2}|[0-9]{3})/,

        tokenizer: {
            root: [
                // Whitespace.
                {include: '@whitespace'},

                // Identifiers and keywords.
                [
                    /@identifier/,
                    {
                        cases: {
                            '@types': 'type',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],

                // Special identified used for the not function.
                [/~/, 'identifier'],

                // Type variable, e.g. 'n
                [/'@identifier/, 'type'],

                // (operator_chars)+, optionally followed by _identifier.
                [/[!%&*+\-./:<>=@^|#]+(?:_@identifier)?/, 'operator'],

                // TODO: Handle < > in $include <foo.h> not being an operator.
                // Although it probably doesn't matter too much that we colour
                // it incorrectly, just for syntax highlighting.

                // Brackets.
                [/[{}()[\]<>]/, '@brackets'],

                // Numbers.
                [/0b[01_]+/, 'number.binary'],
                [/0x[0-9a-fA-F_]+/, 'number.hex'],
                [/-?\d*\.\d+/, 'number.float'],
                [/-?\d+/, 'number'],

                // TODO: is . on it's own a delimiter or an operator?

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // Unterminated string. Any character except " or \,
                // or \ followed by any character. Then end of line.
                // TODO: I think this doesn't mark unclosed multiline
                // strings as invalid.
                [/"([^"\\]|\\.)*$/, 'string.invalid'],
                // Valid string.
                [/"/, 'string', '@string'],
            ],

            whitespace: [
                // Whitespace.
                [/[ \t\r\n]+/, 'white'],
                // Start of block comment.
                [/\/\*/, 'comment', '@block_comment'],
                // Line comment.
                [/\/\/.*$/, 'comment'],
            ],

            block_comment: [
                // Not / or *, definitely still in the block.
                // This is not strictly necessary but improves efficiency.
                [/[^/*]+/, 'comment'],
                // /*, push block comment.
                [/\/\*/, 'comment', '@push'],
                // */, pop block comment.
                ['\\*/', 'comment', '@pop'],
                // Anything else, still a comment.
                [/./, 'comment'],
            ],

            string: [
                // Not \ or " - must still be in the string.
                [/[^\\"]+/, 'string'],
                // Valid escape sequences, including escaped new lines.
                [/@string_escapes/, 'string.escape'],
                // Any other escape sequence is invalid.
                [/\\./, 'string.escape.invalid'],
                // End of string.
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

monaco.languages.register({id: 'sail'});
monaco.languages.setMonarchTokensProvider('sail', definition());
