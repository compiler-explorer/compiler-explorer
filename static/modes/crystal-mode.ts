// Copyright (c) 2021, Compiler Explorer Authors
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

function configuration(): monaco.languages.LanguageConfiguration {
    return {
        comments: {
            lineComment: '#',
        },

        brackets: [
            ['(', ')'],
            ['{', '}'],
            ['[', ']'],
        ],

        autoClosingPairs: [
            {open: '{', close: '}'},
            {open: '[', close: ']'},
            {open: '(', close: ')'},
            {open: '"', close: '"'},
            {open: "'", close: "'"},
        ],

        surroundingPairs: [
            {open: '{', close: '}'},
            {open: '[', close: ']'},
            {open: '(', close: ')'},
            {open: '"', close: '"'},
            {open: "'", close: "'"},
        ],

        indentationRules: {
            /* TODO(supergrecko): Investigate inefficient regex
             *   original regexp was: /^\s*((begin|class|module|struct|union|annotation|lib|(private|protected)\s+(def|macro)|def|macro|else|elsif|ensure|if|rescue|unless|until|when|while|case)|([^#]*\sdo\b)|([^#]*=\s*(case|if|unless|while|until|begin)))\b([^#{;]|(["'\\/]).*\4|{{.*?}}|{.*?})*(#.*)?$/
             *   truncated piece in fixed regex is: \b([^#{;]|(["'\\/]).*\4|{{.*?}}|{.*?})*(#.*)?
             *
             * See https://github.com/github/codeql/blob/06b36f742e88b610e0c2b75e2668603fc3a0f359/javascript/ql/src/Performance/ReDoS.ql for reference
             */
            increaseIndentPattern:
                /^\s*((begin|class|module|struct|union|annotation|lib|(private|protected)\s+(def|macro)|def|macro|else|elsif|ensure|if|rescue|unless|until|when|while|case)|([^#]*\sdo\b)|([^#]*=\s*(case|if|unless|while|until|begin)))$/,
            decreaseIndentPattern: /^\s*([}\]]([,)]?\s*(#|$)|\.[a-zA-Z_]\w*\b)|(end|rescue|ensure|else|elsif|when)\b)/,
        },
    };
}

function definition(): monaco.languages.IMonarchLanguage {
    return {
        keywords: [
            'abstract',
            'alias',
            'annotation',
            'asm',
            'begin',
            'break',
            'case',
            'class',
            'def',
            'do',
            'else',
            'elsif',
            'end',
            'ensure',
            'enum',
            'extend',
            'false',
            'for',
            'forall',
            'fun',
            'if',
            'in',
            'include',
            'instance_sizeof',
            'lib',
            'macro',
            'module',
            'next',
            'nil',
            'offsetof',
            'out',
            'pointerof',
            'private',
            'protected',
            'require',
            'rescue',
            'return',
            'select',
            'self',
            'sizeof',
            'struct',
            'true',
            'type',
            'typeof',
            'uninitialized',
            'union',
            'unless',
            'until',
            'verbatim',
            'when',
            'while',
            'yield',
        ],

        operators: [
            '+',
            '&+',
            '-',
            '&-',
            '!',
            '~',
            '**',
            '&**',
            '*',
            '&*',
            '/',
            '//',
            '%',
            '<<',
            '>>',
            '&',
            '|',
            '^',
            '==',
            '!=',
            '=~',
            '!~',
            '===',
            '<',
            '<=',
            '>',
            '>=',
            '<=>',
            '&&',
            '||',
            '..',
            '...',
            '?',
            ':',
            '=',
            '[]=',
            '+=',
            '&+=',
            '-=',
            '&-=',
            '*=',
            '&*=',
            '/=',
            '//=',
            '%=',
            '|=',
            '&=',
            '^=',
            '**=',
            '<<=',
            '>>=',
            '||=',
            '&&=',
            '[]',
            '[]?',
            '&.',
        ],

        symbols: /[=><!~?:&|+\-*/^%.]+/,
        escapes: /\\(?:[abefnrtv\\"'#]|[0-7]{1,3}|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|u{[0-9A-Fa-f]{1,6}})/,
        numsuffix: /i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|f32|f64/,

        tokenizer: {
            root: [
                [/__(DIR|END_LINE|FILE|LINE)__/, 'keyword'],

                [
                    /[a-z_][a-z0-9_]*[!?=]?/,
                    {
                        cases: {
                            '@keywords': 'keyword',
                            '@default': 'identifier',
                        },
                    },
                ],

                [/[A-Z][A-Za-z0-9_]*/, 'type.identifier'],

                [/[;,.]/, 'delimiter'],
                [/::/, 'delimiter'],

                [
                    /@symbols/,
                    {
                        cases: {
                            '@operators': 'operator',
                            '@default': '',
                        },
                    },
                ],

                [/[{}()[\]]/, '@brackets'],

                [/0x[0-9A-Fa-f_]+(@numsuffix)?/, 'number.hex'],
                [/0o[0-7_]+(@numsuffix)?/, 'number.octal'],
                [/0b[01_]+(@numsuffix)?/, 'number.binary'],
                [/(?!00)[0-9][0-9_]*\.[0-9][0-9_]*([eE][-+]?[0-9][0-9_]*)?(f32|f64)?/, 'number.float'],
                [/(?!00)[0-9][0-9_]*(@numsuffix)?/, 'number'],

                [/#.*$/, 'comment'],

                [/"/, 'string', '@string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'[^\\']'/, 'string'],
            ],

            string: [
                [/@escapes/, 'string.escape'],
                [/"/, 'string', '@pop'],
                [/./, 'string'],
            ],
        },
    };
}

const def = definition();

monaco.languages.register({id: 'crystal'});
monaco.languages.setMonarchTokensProvider('crystal', def);
monaco.languages.setLanguageConfiguration('crystal', configuration());

export = def;
