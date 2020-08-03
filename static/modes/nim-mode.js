// Copyright (c) 2019, Ray Imber 
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
    // Nim language definition
    
    return {
        keywords: [
            'addr', 'as', 'asm',
            'bind', 'block', 'break',
            'case', 'cast', 'concept', 'const', 'continue', 'converter',
            'defer', 'discard', 'distinct', 'div', 'do',
            'elif', 'else', 'end', 'enum', 'except', 'export',
            'finally', 'for', 'from', 'func',
            'if', 'import', 'include', 'interface', 'iterator',
            'let',
            'macro', 'method', 'mixin', 'mod',
            'nil',
            'object', 'out',
            'proc', 'ptr',
            'raise', 'ref', 'return',
            'static',
            'template', 'try', 'tuple', 'type',
            'using',
            'var',
            'when', 'while',
            'yield',
            'push', 'pop',
        ],
        operators: [
            '=', '+', '-', '*', '/', '<', '>',
            '@', '$', '~', '&', '%', '|',
            '!', '?', '^', '.', ':', '\\',
        ],
        wordOperators: [
            'and', 'or', 'not', 'xor', 'shl', 'shr', 'div', 'mod', 'in', 'notin', 'is', 'isnot', 'of',
        ],
        symbols: /[!%&*+/<=>^|~-]+/,
        escapes: /\\(p|r|c|n|l|f|t|v|a|b|e|\\|"|'|\d+|x[\dA-Fa-f]{2}|u[\dA-Fa-f]{4}|u{[\dA-Fa-f]+})/,
        charEscapes: /\\(r|c|n|l|f|t|v|a|b|e|\\|"|'|x[\dA-Fa-f]{2})/,
        
        hexNumber: /0(xX)[\dA-Fa-f](_?[\dA-Fa-f])*/,
        decNumber: /\d(_?\d)*/, 
        octNumber: /0o[0-7](_?[0-7])*/,
        binNumber: /0(bB)[01](_?[01])*/,
        exponent: /(eE)(\+-)?\d(_?\d)*/,
        brackets: [
            ['{','}','delimiter.curly'],
            ['{.','.}','delimiter.curly'],
            ['[',']','delimiter.square'],
            ['[:',']','delimiter.square'],
            ['[.','.]','delimiter.square'],
            ['(',')','delimiter.parenthesis'],
            ['(.','.)','delimiter.parenthesis'],
            ['<','>','delimiter.angle'],
        ],

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                [/[A-Za-z](_?\w)*/, {
                    cases: {
                        '@keywords': 'keyword',
                        '@wordOperators': 'keyword',
                        '@default': 'identifier',
                    },
                }],
                {include: '@whitespace'},
                [/([(:[{|]\.|\.[)\]}]|[()[\]{}])/, '@brackets'],
                [/@symbols/, {
                    cases: {
                        '@operators': 'operator',
                        '@default': '',
                    },
                }],

                // number literals
                // floats
                [/@decNumber(\.@decNumber(@exponent)|@exponent)(')?([DFdf])(32|64)?/, 'number.float'],
                [/(@decNumber|@octNumber|@binNumber)(')?([DFdf])(32|64)?/, 'number.float'],
                [/@hexNumber'([DFdf])(32|64)?/, 'number.float'],

                // ints
                [/@hexNumber(')?(([IUiu])(8|16|32|64))?/, 'number.hex'],
                [/@octNumber(')?(([IUiu])(8|16|32|64))?/, 'number.octal'],
                [/@binNumber(')?(([IUiu])(8|16|32|64))?/, 'number.binary'],

                [/@decNumber(')?(([IUiu])(8|16|32|64))?/, 'number'],
              
                // char literals
                [/'/, 'string', '@character'],

                // strings
                [/(r|R)"/, 'string', '@rawString'],
                [/"""/, 'string', '@tripleQuoteString'],
                [/"(?!")/, 'string', '@string'],
            ],
            whitespace: [
                [/[\t\n\r ]+/, 'white'],
                [/#\[/, 'comment', '@comment'],
                [/#.*$/, 'comment'],
            ],
            comment: [
                [/[^#\]]/, 'comment'],
                [/]#/, 'comment', '@pop'],
            ],
            string: [
                [/@escapes/, 'string.escape'],
                [/"/, 'string', '@pop'],
            ],
            tripleQuoteString: [
                [/"""/, 'string', '@pop'],
            ],
            rawString: [
                [/"/, 'string', '@pop'],
            ],
            character: [
                [/@charEscapes/, 'string.escape'],
                [/'/, 'string', '@pop'],
            ],
        },
    };
}

monaco.languages.register({id: 'nim'});
monaco.languages.setMonarchTokensProvider('nim', definition());
