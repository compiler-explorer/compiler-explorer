// Copyright (c) 2022, Compiler Explorer Authors
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
import * as cpp from 'monaco-editor/esm/vs/basic-languages/cpp/cpp';
import * as cppp from './cppp-mode.js';

function definition(): monaco.languages.IMonarchLanguage {
    const cppfront = $.extend(true, {}, cppp); // deep copy
    cppfront.tokenPostfix = '.herb';

    // Ideally, all this only works in Cpp2 syntax only.
    // (Cpp2 can be interleaved with Cpp1 at top level declarations).

    // Adapted from `cppp-mode.ts`.
    function removeKeyword(keyword) {
        const index = cppfront.keywords.indexOf(keyword);
        if (index > -1) {
            cppfront.keywords.splice(index, 1);
        }
    }

    function removeKeywords(keywords) {
        for (let i = 0; i < keywords.length; ++i) {
            removeKeyword(keywords[i]);
        }
    }

    // Reclaimed identifiers (<https://github.com/hsutter/cppfront/blob/2c64707179a6c961b0592b6411dae8bf4c4a85d0/source/cppfront.cpp#L1543>).
    removeKeywords([
        'and',
        'and_eq',
        'bitand',
        'bitor',
        'compl',
        'not',
        'not_eq',
        'or',
        'or_eq',
        'xor',
        'xor_eq',
        'new',
        'class',
        'struct',
    ]);

    removeKeywords(['override', 'final']);

    // Cpp2 keywords.
    cppfront.keywords.push('i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'inspect');

    // Cpp2 contextual keywords.

    // Ideally, these are only highlighted in the context they are keywords.
    cppfront.keywords.push('in', 'inout', 'copy', 'out', 'move', 'forward', 'next', 'is', 'as');

    // _this-specifier_
    cppfront.tokenizer.root.unshift([/(implicit|override|final)(?=\s+this\b)/, 'keyword']);

    // _throws-specifier_
    cppfront.tokenizer.throws_specifier = [[/throws/, 'keyword', 'root']];
    cppfront.tokenizer.root.unshift([/\)\s*(?=throws)/, '', 'throws_specifier']);

    // _contract-kind_
    cppfront.tokenizer.contract_kind = [[/pre|post|assert/, 'keyword', 'root']];
    cppfront.tokenizer.root.unshift([/\[\s*\[\s*(?=pre|post|assert)/, '', 'contract_kind']);

    // `final`
    cppfront.tokenizer.root.unshift([/final(?=\s+type\b)/, 'keyword']);
    // `type`
    cppfront.tokenizer.root.unshift([/type(?=(\s+requires\b|\s*=))/, 'keyword']);

    // Identifiers with special meaning could use some highlighting:
    // `finally`:
    // https://github.com/hsutter/cppfront/blob/472bf58d74d2fba4b09d38894379483b17741844/source/cppfront.cpp#L1526
    // `unique.new`, `shared.new`:
    // https://github.com/hsutter/cppfront/blob/472bf58d74d2fba4b09d38894379483b17741844/source/cppfront.cpp#L1708
    // Predefined contract groups:
    // https://github.com/hsutter/cppfront/blob/472bf58d74d2fba4b09d38894379483b17741844/source/cppfront.cpp#L4053

    cppfront.tokenizer.root.unshift([/(?:[a-zA-Z_]\w*\s*)?:(?!:)/, 'identifier.definition']);

    return cppfront;
}

monaco.languages.register({id: 'cpp2-cppfront'});
monaco.languages.setLanguageConfiguration('cpp2-cppfront', cpp.conf);
monaco.languages.setMonarchTokensProvider('cpp2-cppfront', definition());

export {};
