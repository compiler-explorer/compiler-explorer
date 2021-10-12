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
import _ from 'underscore';
import {
    conf as BaseCppConfig,
    language as BaseCppLanguage,
} from 'monaco-editor/esm/vs/basic-languages/cpp/cpp';

// We remove everything that's not an identifier, underscore reserved name and not an official C++ keyword...
// Regarding #617, final is a identifier with special meaning, not a fully qualified keyword
const KEYWORDS_TO_REMOVE = ['abstract', 'amp', 'array', 'cpu', 'delegate', 'each', 'event', 'finally', 'gcnew',
    'generic', 'in', 'initonly', 'interface', 'interior_ptr', 'internal', 'literal', 'partial', 'pascal',
    'pin_ptr', 'property', 'ref', 'restrict', 'safe_cast', 'sealed', 'title_static', 'where'];

const KEYWORDS_TO_ADD = ['alignas', 'alignof', 'and', 'and_eq', 'asm', 'bitand', 'bitor', 'char8_t', 'char16_t',
    'char32_t', 'compl', 'concept', 'consteval', 'constinit', 'co_await', 'co_return', 'co_yield', 'not', 'not_eq',
    'or', 'or_eq', 'requires', 'xor', 'xor_eq'];

// We need to create a new definition for cpp so we can remove invalid keywords
export const definition: () => monaco.languages.IMonarchLanguage = () => _.extend(BaseCppLanguage, {
    keywords: BaseCppLanguage.keywords
        .filter((kw => !KEYWORDS_TO_REMOVE.includes(kw)))
        .concat(KEYWORDS_TO_ADD),
});

monaco.languages.register({id: 'cppp'});
monaco.languages.setMonarchTokensProvider('cppp', definition());
monaco.languages.setLanguageConfiguration('cppp', BaseCppConfig);
