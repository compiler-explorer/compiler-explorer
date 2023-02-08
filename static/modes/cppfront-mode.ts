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
import * as cppp from './cppp-mode';

function definition(): monaco.languages.IMonarchLanguage {
    const cppfront = $.extend(true, {}, cppp); // deep copy
    cppfront.tokenPostfix = '.herb';

    cppfront.keywords.push('type', 'next', 'inout');

    // pick up 'identifier:' as a definition. This is a hack; ideally we'd have "optional whitespace after beginning of
    // line" but here I just try to disambiguate `::` and don't worry about labels.
    cppfront.tokenizer.root.unshift([/[a-zA-Z_]\w*\s*:(?!:)/, 'identifier.definition']);

    return cppfront;
}

monaco.languages.register({id: 'cpp2-cppfront'});
monaco.languages.setLanguageConfiguration('cpp2-cppfront', cpp.conf);
monaco.languages.setMonarchTokensProvider('cpp2-cppfront', definition());

export {};
