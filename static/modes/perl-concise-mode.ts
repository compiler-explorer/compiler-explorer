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
//
// Some typical "concise" output:
// 2     <;> nextstate(main 48 example.pl:2) v:U,{ ->3
// 5     <@> binmode vK/2 ->6
// -        <0> ex-pushmark s ->3
// 3        <#> gv[*STDOUT] s ->4
//
// components are:
// label|- \s+ <opclass> \s+ "ex-"?opcode optparams optarg opttargarg flags/optprivateflags hints ->next
import * as monaco from 'monaco-editor';

function definition(): monaco.languages.IMonarchLanguage {
    return {
        // Set defaultToken to invalid to see what you do not tokenize yet
        defaultToken: 'invalid',

        // concise has a limited set of escapes it produces
        // but that may change
        escapes: /\\(?:[abfnrtv\\"']|[0-7]{3})/,

        tokenizer: {
            root: [
                // Generated label
                [/^([a-zA-Z0-9]+)(?=\s)/, {token: 'identifier', next: '@opclass'}],

                // unlabelled line, typically START op or nulled op
                // not really a comment
                [/^-(?=\s)/, {token: 'comment', next: '@opclass'}],

                // function header
                [/^([^:]+)(?=:$)/, {token: 'type.identifier'}],

                {include: '@whitespace'},
            ],

            opclass: [
                // pop at the beginning of the next line and rematch
                [/^.*$/, {token: '@rematch', next: '@pop'}],

                // opclass
                [/<[012|@/$#{%}\-";.+]>(?=\s)/, {token: 'comment', next: '@opcode'}],

                {include: '@whitespace'},
            ],

            opcode: [
                // pop at the beginning of the next line and rematch
                [/^.*$/, {token: '@rematch', next: '@pop'}],

                // opcode
                // immediately after the opclass, possible ex- prefix
                [/((?:ex-)?)((?!ex-)\w+)/, ['comment', {token: 'keyword', next: '@rest'}]],
            ],

            rest: [
                // pop at the beginning of the next line and rematch
                [/^.*$/, {token: '@rematch', next: '@pop'}],

                {include: '@whitespace'},
            ],

            whitespace: [[/[ \t\r\n]+/, 'white']],
        },
    };
}

const def = definition();
monaco.languages.register({id: 'perl-concise'});
monaco.languages.setMonarchTokensProvider('perl-concise', def);

export default def;
