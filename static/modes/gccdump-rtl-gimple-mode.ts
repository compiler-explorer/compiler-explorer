// Copyright (c) 2017, Marc Poulhiès - Kalray Inc.
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Released under the MIT license

// this is mostly based on 'mylang' example from https://microsoft.github.io/monaco-editor/monarch.html
import * as monaco from 'monaco-editor';

function definition(): monaco.languages.IMonarchLanguage {
    return {
        // Set defaultToken to invalid to see what you do not tokenize yet
        // defaultToken: 'invalid',

        keywords: [
            'abstract',
            'continue',
            'for',
            'new',
            'switch',
            'assert',
            'goto',
            'do',
            'if',
            'private',
            'this',
            'break',
            'protected',
            'throw',
            'else',
            'public',
            'enum',
            'return',
            'catch',
            'try',
            'interface',
            'static',
            'class',
            'finally',
            'const',
            'super',
            'while',
            'true',
            'false',

            // Generated using the following:
            // #define DEF_RTL_EXPR(a,b,c,d) b,
            // keyword: [
            // #include <rtl.def>
            // ],
            // And by invoking the cpp :
            // cpp -P -I/path/to/gcc/gcc/

            // All RTL classes.
            'UnKnown',
            'value',
            'debug_expr',
            'expr_list',
            'insn_list',
            'int_list',
            'sequence',
            'address',
            'debug_insn',
            'insn',
            'jump_insn',
            'call_insn',
            'jump_table_data',
            'barrier',
            'code_label',
            'note',
            'cond_exec',
            'parallel',
            'asm_input',
            'asm_operands',
            'unspec',
            'unspec_volatile',
            'addr_vec',
            'addr_diff_vec',
            'prefetch',
            'set',
            'use',
            'clobber',
            'call',
            'return',
            'simple_return',
            'eh_return',
            'trap_if',
            'const_int',
            'const_fixed',
            'const_double',
            'const_vector',
            'const_string',
            'const',
            'pc',
            'reg',
            'scratch',
            'subreg',
            'strict_low_part',
            'concat',
            'concatn',
            'mem',
            'label_ref',
            'symbol_ref',
            'cc0',
            'if_then_else',
            'compare',
            'plus',
            'minus',
            'neg',
            'mult',
            'ss_mult',
            'us_mult',
            'div',
            'ss_div',
            'us_div',
            'mod',
            'udiv',
            'umod',
            'and',
            'ior',
            'xor',
            'not',
            'ashift',
            'rotate',
            'ashiftrt',
            'lshiftrt',
            'rotatert',
            'smin',
            'smax',
            'umin',
            'umax',
            'pre_dec',
            'pre_inc',
            'post_dec',
            'post_inc',
            'pre_modify',
            'post_modify',
            'ne',
            'eq',
            'ge',
            'gt',
            'le',
            'lt',
            'geu',
            'gtu',
            'leu',
            'ltu',
            'unordered',
            'ordered',
            'uneq',
            'unge',
            'ungt',
            'unle',
            'unlt',
            'ltgt',
            'sign_extend',
            'zero_extend',
            'truncate',
            'float_extend',
            'float_truncate',
            'float',
            'fix',
            'unsigned_float',
            'unsigned_fix',
            'fract_convert',
            'unsigned_fract_convert',
            'sat_fract',
            'unsigned_sat_fract',
            'abs',
            'sqrt',
            'bswap',
            'ffs',
            'clrsb',
            'clz',
            'ctz',
            'popcount',
            'parity',
            'sign_extract',
            'zero_extract',
            'high',
            'lo_sum',
            'vec_merge',
            'vec_select',
            'vec_concat',
            'vec_duplicate',
            'ss_plus',
            'us_plus',
            'ss_minus',
            'ss_neg',
            'us_neg',
            'ss_abs',
            'ss_ashift',
            'us_ashift',
            'us_minus',
            'ss_truncate',
            'us_truncate',
            'fma',
            'var_location',
            'debug_implicit_ptr',
            'entry_value',
            'debug_parameter_ref',
        ],

        typeKeywords: ['boolean', 'double', 'byte', 'int', 'short', 'char', 'void', 'long', 'float'],

        operators: [
            '=',
            '>',
            '<',
            '!',
            '~',
            '?',
            ':',
            '==',
            '<=',
            '>=',
            '!=',
            '&&',
            '||',
            '++',
            '--',
            '+',
            '-',
            '*',
            '/',
            '&',
            '|',
            '^',
            '%',
            '<<',
            '>>',
            '>>>',
            '+=',
            '-=',
            '*=',
            '/=',
            '&=',
            '|=',
            '^=',
            '%=',
            '<<=',
            '>>=',
            '>>>=',
        ],

        // we include these common regular expressions
        symbols: /[=><!~?:&|+\-*/^%]+/,

        // C# style strings
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        // The main tokenizer for our languages
        tokenizer: {
            root: [
                // identifiers and keywords
                [
                    /[a-z_$][\w$]*/,
                    {
                        cases: {
                            '@typeKeywords': 'keyword',
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],
                [/[A-Z][\w$]*/, 'type.identifier'], // to show class names nicely

                // whitespace
                {include: '@whitespace'},

                // delimiters and operators
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

                // @ annotations.
                // As an example, we emit a debugging log message on these tokens.
                // Note: message are supressed during the first load -- change some lines to see them.
                //  [/@\s*[a-zA-Z_\$][\w\$]*/, { token: 'annotation', log: 'annotation token: $0' }],

                // numbers
                [/\d*\.\d+([eE][-+]?\d+)?/, 'number.float'],
                [/0[xX][0-9a-fA-F]+/, 'number.hex'],
                [/\d+/, 'number'],

                // delimiter: after number because of .\d floats
                [/[;,.]/, 'delimiter'],

                // strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                [/"/, {token: 'string.quote', bracket: '@open', next: '@string'}],

                // characters
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid'],
            ],

            comment: [
                [/[^/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'], // nested comment
                ['\\*/', 'comment', '@pop'],
                [/[/*]/, 'comment'],
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, {token: 'string.quote', bracket: '@close', next: '@pop'}],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment'],
                [/^;;.*$/, 'comment'],
            ],
        },
    };
}

monaco.languages.register({id: 'gccdump-rtl-gimple'});
monaco.languages.setMonarchTokensProvider('gccdump-rtl-gimple', definition());

export {};
