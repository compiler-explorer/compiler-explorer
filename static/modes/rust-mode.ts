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

// @ts-ignore  "Could not find a declaration file"
import * as rust from 'monaco-editor/esm/vs/basic-languages/rust/rust';

// We need to patch the existing rust definition to fix hexadecimal literal highlighting
// This fixes a bug where hex literals with underscores after hex letters (like 0x01_02_0a_0b)
// lose highlighting after the underscore that follows hex letters (a-f).

function definition(): monaco.languages.IMonarchLanguage {
    const rustPatched = $.extend(true, {}, rust.language); // deep copy

    // Fix the hexadecimal pattern in the numbers tokenizer
    // Original Monaco pattern: /(0x[\da-fA-F]+)_?(@intSuffixes)?/
    // Fixed pattern: /(0x[0-9a-fA-F_]+)(@intSuffixes)?/
    // This allows underscores throughout the hex number, not just at the end
    rustPatched.tokenizer.numbers[4] = [/(0x[0-9a-fA-F_]+)(@intSuffixes)?/, {token: 'number'}];

    return rustPatched;
}

const def = definition();

monaco.languages.register({id: 'rustp'});
monaco.languages.setLanguageConfiguration('rustp', rust.conf);
monaco.languages.setMonarchTokensProvider('rustp', def);

export default def;
