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

// References:
// https://github.com/microsoft/monaco-editor/blob/main/src/languages/definitions/python/python.ts
// https://mojolang.org/docs/reference/
// https://mojolang.org/docs/manual/
function definition(): monaco.languages.IMonarchLanguage {
    return {
        defaultToken: '',
        tokenPostfix: '.mojo',

        keywords: [
            // Python-inherited keywords
            'False',
            'None',
            'True',
            'and',
            'as',
            'assert',
            'async',
            'await',
            'break',
            'class',
            'continue',
            'def',
            'del',
            'elif',
            'else',
            'except',
            'finally',
            'for',
            'from',
            'global',
            'if',
            'import',
            'in',
            'is',
            'lambda',
            'nonlocal',
            'not',
            'or',
            'pass',
            'raise',
            'return',
            'self',
            'try',
            'while',
            'with',
            'yield',

            // Mojo declarations
            'fn',       // legacy (kept for older compilers)
            'struct',
            'trait',
            'var',
            'ref',
            'comptime', // current spelling of compile-time declarations
            'alias',    // pre-comptime spelling
            'let',      // legacy

            // Mojo ownership
            'read',
            'mut',
            'out',
            'deinit',
            'owned',    // old spelling of `var`
            'borrowed', // old spelling of `read`
            'inout',    // old spelling of `mut`

            // Mojo functions
            'raises',
            'capturing',
            'escaping',
            'unified',
            'where',

            // Mojo MLIR-level primitives
            '__mlir_attr',
            '__mlir_op',
            '__mlir_type',
        ],

        // Common standard-library types
        // (not exhaustive)
        typeKeywords: [
            'AnyType',
            'Bool',
            'Byte',
            'DType',
            'Dict',
            'Error',
            'Float16',
            'Float32',
            'Float64',
            'Int',
            'Int128',
            'Int16',
            'Int256',
            'Int32',
            'Int64',
            'Int8',
            'List',
            'NoneType',
            'Optional',
            'Pointer',
            'SIMD',
            'Scalar',
            'Self',
            'Set',
            'Span',
            'String',
            'StringLiteral',
            'StringSlice',
            'Tuple',
            'UInt',
            'UInt128',
            'UInt16',
            'UInt256',
            'UInt32',
            'UInt64',
            'UInt8',
            'UnsafePointer',
            'object',
        ],

        // Python-inspired
        brackets: [
            { open: '{', close: '}', token: 'delimiter.curly' },
            { open: '[', close: ']', token: 'delimiter.bracket' },
            { open: '(', close: ')', token: 'delimiter.parenthesis' }
        ],

        tokenizer: {
            root: [
                {include: '@whitespace'},
                {include: '@numbers'},
                {include: '@strings'},
            ],

            // Comments can be anywhere on a line
            whitespace: [
                [/\s+/, 'white'],
                [/#.*$/, 'comment'],
            ],

            numbers: [
                [/0[xX][0-9a-fA-F](_?[0-9a-fA-F])*/, 'number.hex'],
                [/0[bB][01](_?[01])*/, 'number.binary'],
                [/0[oO][0-7](_?[0-7])*/, 'number.octal'],
                [/\d(_?\d)*\.\d(_?\d)*([eE][-+]?\d(_?\d)*)?/, 'number.float'],
                [/\.\d(_?\d)*([eE][-+]?\d(_?\d)*)?/, 'number.float'],
                [/\d(_?\d)*[eE][-+]?\d(_?\d)*/, 'number.float'],
                [/\d(_?\d)*/, 'number'],
            ],

            // Python prefixes (bBfFrRuU) + Mojo prefix (template)
            strings: [
                [/[bBfFrRtTuU]{0,2}'''/, 'string', '@tripleSingle'],
                [/[bBfFrRtTuU]{0,2}"""/, 'string', '@tripleDouble'],
                [/[bBfFrRtTuU]{0,2}'/, 'string.escape', '@stringBody'],
                [/[bBfFrRtTuU]{0,2}"/, 'string.escape', '@dblStringBody'],
            ],
            tripleSingle: [
                [/[^\\']+/, 'string'],
                [/\\./, 'string.escape'],
                [/'''/, 'string', '@popall'],
                [/'/, 'string'],
            ],
            tripleDouble: [
                [/[^\\"]+/, 'string'],
                [/\\./, 'string.escape'],
                [/"""/, 'string', '@popall'],
                [/"/, 'string'],
            ],
            stringBody: [
                [/[^\\']+$/, 'string', '@popall'],
                [/[^\\']+/, 'string'],
                [/\\./, 'string.escape'],
                [/'/, 'string.escape', '@popall'],
                [/\\$/, 'string'],
            ],
            dblStringBody: [
                [/[^\\"]+$/, 'string', '@popall'],
                [/[^\\"]+/, 'string'],
                [/\\./, 'string.escape'],
                [/"/, 'string.escape', '@popall'],
                [/\\$/, 'string'],
            ],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'mojo'});
monaco.languages.setMonarchTokensProvider('mojo', def);
export default def;
