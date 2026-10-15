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

import $ from 'jquery';
import * as monaco from 'monaco-editor';
import * as ruby from 'monaco-editor/esm/vs/basic-languages/ruby/ruby';

// Hopefully temporary: upstream's Ruby grammar throws "trying to pop an empty stack" on valid code such as
// `x = if a ... end`, because it only opens a block for `if` at the start of a statement, so the `end` pops the
// root state (https://github.com/microsoft/monaco-editor/issues/134). This ports the fix from
// https://github.com/microsoft/monaco-editor/pull/5509; drop it, and the `rubyp` id, once that lands in our
// monaco-editor.

const issueHint = 'check if https://github.com/microsoft/monaco-editor/issues/134 is fixed upstream';

function patchUnmatchedEnd(rules: monaco.languages.IMonarchLanguageRule[]): number {
    let patched = 0;
    for (const rule of rules) {
        if (!Array.isArray(rule)) continue;
        const actions = Array.isArray(rule[1]) ? rule[1] : [rule[1]];
        for (const action of actions) {
            if (typeof action !== 'object') continue;
            const cases = action.cases as Record<string, monaco.languages.IExpandedMonarchLanguageAction> | undefined;
            if (cases?.end?.next === '@pop') {
                // Only pop when a block is open; a stray `end` is just a keyword.
                cases.end = {cases: {$S2: {token: 'keyword.$S2', next: '@pop'}, '@default': 'keyword'}};
                patched++;
            }
        }
    }
    return patched;
}

function definition(): monaco.languages.IMonarchLanguage {
    const rubyPatched = $.extend(true, {}, ruby.language); // deep copy
    const tokenizer = rubyPatched.tokenizer;

    const endRules = ['root', 'dodecl', 'modifier'].reduce(
        (count, state) => count + patchUnmatchedEnd(tokenizer[state]),
        0,
    );
    if (endRules !== 4) {
        throw new Error(`Monaco Ruby: expected 4 'end' rules to patch, found ${endRules} - ${issueHint}`);
    }

    const brackets = tokenizer.root.findIndex(
        (rule: monaco.languages.IMonarchLanguageRule) =>
            Array.isArray(rule) && rule[0] instanceof RegExp && rule[0].source === String.raw`[{}()\[\]]`,
    );
    if (brackets === -1) {
        throw new Error(`Monaco Ruby: brackets rule not found - ${issueHint}`);
    }
    // A conditional or loop after an operator, opening bracket or separator is an expression that opens a block,
    // unlike a trailing modifier such as `return if x`.
    tokenizer.root.splice(brackets, 0, [
        /([=><!~?:&|+\-*/^%]+(?=\s)|[([{,;])(\s*)(if|unless|while|until)\b(?![:?!])/,
        [
            {
                cases: {
                    '@keywordops': 'keyword',
                    '@operators': 'operator',
                    '~[(\\[{]': '@brackets',
                    '~[,;]': 'delimiter',
                    '@default': '',
                },
            },
            '',
            {
                cases: {
                    'while|until': {token: 'keyword.$3', next: '@dodecl.$3'},
                    '@default': {token: 'keyword.$3', next: '@root.$3'},
                },
            },
        ],
    ]);

    return rubyPatched;
}

monaco.languages.register({id: 'rubyp'});
monaco.languages.setLanguageConfiguration('rubyp', ruby.conf);
monaco.languages.setMonarchTokensProvider('rubyp', definition());
