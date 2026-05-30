// Copyright (c) 2019, Compiler Explorer Authors
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

// Based on https://github.com/leanprover/vscode-lean4/blob/bbe3ff12/vscode-lean4/syntaxes/lean4.json.
function definition(): monaco.languages.IMonarchLanguage {
    return {
        escapes: /\\(?:[\\'"ntr]|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4})/,
        keywords: [
            'local',
            'scoped',
            'partial',
            'unsafe',
            'nonrec',
            'public',
            'private',
            'protected',
            'noncomputable',
            'meta',
            'deriving',
            'instance',
            'inductive',
            'coinductive',
            'structure',
            'theorem',
            'axiom',
            'abbrev',
            'lemma',
            'def',
            'instance',
            'class',
            'with',
            'extends',
            'where',
            'theorem',
            'show',
            'have',
            'using',
            'haveI',
            'from',
            'suffices',
            'nomatch',
            'nofun',
            'no_index',
            'def',
            'class',
            'structure',
            'instance',
            'elab',
            'set_option',
            'initialize',
            'builtin_initialize',
            'example',
            'inductive_fixpoint',
            'inductive',
            'coinductive_fixpoint',
            'coinductive',
            'termination_by?',
            'termination_by',
            'decreasing_by',
            'partial_fixpoint',
            'axiom',
            'universe',
            'variable',
            'module',
            'import all',
            'import',
            'open',
            'export',
            'prelude',
            'renaming',
            'hiding',
            'do',
            'by?',
            'by',
            'let',
            'letI',
            'let_expr',
            'extends',
            'mutual',
            'mut',
            'where',
            'rec',
            'declare_syntax_cat',
            'syntax',
            'macro_rules',
            'macro',
            'max_prec',
            'leading_parser',
            'elab_rules',
            'deriving',
            'fun',
            'section',
            'namespace',
            'end',
            'prefix',
            'postfix',
            'infixl',
            'infixr',
            'infix',
            'notation',
            'abbrev',
            'if',
            'bif',
            'then',
            'else',
            'calc',
            'matches',
            'match_expr',
            'match',
            'with',
            'forall',
            'for',
            'while',
            'repeat',
            'unless',
            'until',
            'panic!',
            'unreachable!',
            'assert!',
            'try',
            'catch',
            'finally',
            'return',
            'continue',
            'break',
            'exists',
            'mod_cast',
            'include_str',
            'include',
            'in',
            'trailing_parser',
            'tactic_tag',
            'tactic_alt',
            'tactic_extension',
            'register_tactic_tag',
            'binder_predicate',
            'grind_propagator',
            'builtin_grind_propagator',
            'grind_pattern',
            'simproc',
            'builtin_simproc',
            'simproc_decl',
            'builtin_simproc_decl',
            'dsimproc',
            'builtin_dsimproc',
            'dsimproc_decl',
            'builtin_dsimproc_decl',
            'show_panel_widgets',
            'show_term',
            'seal',
            'unseal',
            'nat_lit',
            'norm_cast_add_elim',
            'println!',
            'declare_config_elab',
            'register_error_explanation',
            'register_builtin_option',
            'register_option',
            'register_parser_alias',
            'register_simp_attr',
            'register_linter_set',
            'register_label_attr',
            'recommended_spelling',
            'reportIssue!',
            'reprove',
            'run_elab',
            'run_cmd',
            'run_meta',
            'add_decl_doc',
            'omit',
            'opaque',
            'dbg_trace',
            'trace_goal',
            'trace',
            'throwErrorAt',
            'throwError',
            'throwNamedErrorAt',
            'throwNamedError',
            'logNamedWarningAt',
            'logNamedWarning',
            'logNamedErrorAt',
            'logNamedError',
        ],
        types: ['Prop', 'Type', 'Sort'],
        invalidKeywords: ['sorry', 'admit'],
        commands: [
            '#print',
            '#eval',
            '#eval!',
            '#reduce',
            '#synth',
            '#widget',
            '#where',
            '#version',
            '#with_exporting',
            '#check',
            '#check_tactic',
            '#check_tactic_failure',
            '#check_failure',
            '#check_simp',
            '#discr_tree_key',
            '#discr_tree_simp_key',
            '#guard',
            '#guard_expr',
            '#guard_msgs',
        ],
        invalidCommands: ['#exit'],

        tokenizer: {
            root: [
                [/r(#*)"/, {token: 'string.quote', bracket: '@open', next: '@stringraw.$1'}],
                {include: '@whitespace'},

                [/"/, 'string', '@string'],
                [/(?<![\]\w])'[^\\']'/, 'string'],
                [/(?<![\]\w])(')(@escapes)(')/, ['string', 'string.escape', 'string']],

                [/\b(0[xX][0-9a-fA-F]+|[0-9]+|-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?)\b/, 'number'],
                [/«/, 'identifier', '@escapedIdentifier'],
                [/\battribute\b\s*\[[^\]\s]*\]/, 'keyword.modifier'],
                [/@\[[^\]\s]*\]/, 'keyword.modifier'],
                [
                    /#[a-zA-Z_][a-zA-Z0-9_!]*/,
                    {
                        cases: {
                            '@invalidCommands': 'invalid',
                            '@commands': 'keyword',
                            '@default': '',
                        },
                    },
                ],
                [/[{}()[\]]/, '@brackets'],
                [
                    /[a-zA-Z_][a-zA-Z0-9_'?!]*/,
                    {
                        cases: {
                            '@invalidKeywords': 'invalid',
                            '@keywords': 'keyword',
                            '@types': 'keyword.type',
                            '@default': 'identifier',
                        },
                    },
                ],
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/--.*$/, 'comment'],
                [/\/--/, 'comment.doc', '@docComment'],
                [/\/-!/, 'comment.doc', '@docComment'],
                [/\/-/, 'comment', '@blockComment'],
            ],

            docComment: [
                [/[^/-]+/, 'comment.doc'],
                [/\/-/, 'comment.doc', '@blockComment'],
                [/-\//, 'comment.doc', '@pop'],
                [/[/-]/, 'comment.doc'],
            ],

            blockComment: [
                [/[^/-]+/, 'comment'],
                [/\/-/, 'comment', '@push'],
                [/-\//, 'comment', '@pop'],
                [/[/-]/, 'comment'],
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, 'string', '@pop'],
            ],

            escapedIdentifier: [
                [/[^»]+/, 'identifier'],
                [/»/, 'identifier', '@pop'],
            ],

            stringraw: [
                [/[^"#]+/, {token: 'string'}],
                [
                    /"(#*)/,
                    {
                        cases: {
                            '$1==$S2': {token: 'string.quote', bracket: '@close', next: '@pop'},
                            '@default': {token: 'string'},
                        },
                    },
                ],
                [/["#]/, {token: 'string'}],
            ],
        },
    };
}

monaco.languages.register({id: 'lean'});
monaco.languages.setMonarchTokensProvider('lean', definition());
monaco.languages.setLanguageConfiguration('lean', {
    comments: {
        lineComment: '--',
        blockComment: ['/-', '-/'],
    },

    brackets: [
        ['(', ')'],
        ['[', ']'],
        ['{', '}'],
    ],

    autoClosingPairs: [
        {open: '(', close: ')'},
        {open: '[', close: ']'},
        {open: '{', close: '}'},
        {open: '"', close: '"', notIn: ['string', 'comment']},
    ],

    surroundingPairs: [
        {open: '(', close: ')'},
        {open: '[', close: ']'},
        {open: '{', close: '}'},
        {open: '"', close: '"'},
    ],
});
