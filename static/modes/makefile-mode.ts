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
        // Everything make treats as special enough to name: the directives, and the automatic variables a recipe
        // refers to its own target and prerequisites by.
        directives: [
            'define',
            'else',
            'endef',
            'endif',
            'export',
            'ifdef',
            'ifeq',
            'ifndef',
            'ifneq',
            'include',
            'override',
            'private',
            'sinclude',
            'undefine',
            'unexport',
            'vpath',
            '-include',
        ],
        functions: [
            'abspath',
            'addprefix',
            'addsuffix',
            'basename',
            'call',
            'dir',
            'error',
            'eval',
            'file',
            'filter',
            'filter-out',
            'findstring',
            'firstword',
            'foreach',
            'if',
            'info',
            'join',
            'lastword',
            'notdir',
            'origin',
            'patsubst',
            'realpath',
            'shell',
            'sort',
            'strip',
            'subst',
            'suffix',
            'value',
            'warning',
            'wildcard',
            'word',
            'wordlist',
            'words',
        ],
        tokenizer: {
            root: [
                [/#.*$/, 'comment'],
                // A recipe is what follows a tab, and is shell rather than make.
                [/^\t.*$/, 'string'],
                [/^[.\w%][^:=]*(?=::?[^=])/, {cases: {'@default': 'type.identifier'}}],
                [/^\s*([A-Za-z_.][\w.]*)\s*(?=[:+?!]?=)/, 'variable'],
                [/^\s*[-\w]+\b/, {cases: {'@directives': 'keyword', '@default': ''}}],
                {include: '@expansion'},
                [/[:=+?!]/, 'operator'],
                [/"/, 'string', '@string'],
            ],
            // $(foo ...) and ${foo ...}, whose head is a function when make knows the name.
            expansion: [
                [/\$[@<^+*?%]/, 'variable.predefined'],
                [/\$[({]\s*([-\w]+)(?=[\s)}])/, {cases: {'$1@functions': 'keyword', '@default': 'variable'}}],
                [/\$[({]/, 'variable'],
                [/[)}]/, 'variable'],
            ],
            string: [
                [/[^\\"$]+/, 'string'],
                {include: '@expansion'},
                [/\\./, 'string.escape'],
                [/"/, 'string', '@pop'],
            ],
        },
    };
}

function configuration(): monaco.languages.LanguageConfiguration {
    return {
        comments: {lineComment: '#'},
        brackets: [
            ['(', ')'],
            ['{', '}'],
        ],
        autoClosingPairs: [
            {open: '(', close: ')'},
            {open: '{', close: '}'},
            {open: '"', close: '"', notIn: ['string']},
        ],
    };
}

monaco.languages.register({id: 'makefile'});
monaco.languages.setMonarchTokensProvider('makefile', definition());
monaco.languages.setLanguageConfiguration('makefile', configuration());
