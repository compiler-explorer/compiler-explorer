// Copyright (c) 2025, Compiler Explorer Team
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

import * as rust from 'monaco-editor/esm/vs/basic-languages/rust/rust';

// We need to patch the existing rust definition to fix hexadecimal literal highlighting
// This fixes a bug where hex literals with underscores after hex letters (like 0x01_02_0a_0b)
// lose highlighting after the underscore that follows hex letters (a-f).

function definition(): monaco.languages.IMonarchLanguage {
    const rustPatched = $.extend(true, {}, rust.language); // deep copy

    // Fix the hexadecimal pattern in the numbers tokenizer
    // Find and replace the specific problematic pattern to make this robust against Monaco updates
    const expectedPattern = /(0x[\da-fA-F]+)_?(@intSuffixes)?/;
    const fixedPattern = [/(0x[0-9a-fA-F_]+)(@intSuffixes)?/, {token: 'number'}];

    let patternFound = false;
    for (let i = 0; i < rustPatched.tokenizer.numbers.length; i++) {
        const rule = rustPatched.tokenizer.numbers[i];
        if (Array.isArray(rule) && rule[0] instanceof RegExp && rule[0].toString() === expectedPattern.toString()) {
            rustPatched.tokenizer.numbers[i] = fixedPattern;
            patternFound = true;
            break;
        }
    }

    if (!patternFound) {
        throw new Error(
            'Monaco Rust hex pattern not found - check if upstream fixed the issue (https://github.com/microsoft/monaco-editor/issues/4917), remove this patch and revert #7819',
        );
    }

    return rustPatched;
}

const def = definition();

monaco.languages.register({id: 'rustp'});
monaco.languages.setLanguageConfiguration('rustp', rust.conf);
monaco.languages.setMonarchTokensProvider('rustp', def);

export default def;
