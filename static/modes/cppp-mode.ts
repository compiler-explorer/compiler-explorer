// Copyright (c) 2017, Compiler Explorer Authors
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

'use strict';
var monaco = require('monaco-editor');
var cpp = require('monaco-editor/esm/vs/basic-languages/cpp/cpp');

// We need to create a new definition for cpp so we can remove invalid keywords

function definition() {
    var cppp = $.extend(true, {}, cpp.language); // deep copy

    function removeKeyword(keyword) {
        var index = cppp.keywords.indexOf(keyword);
        if (index > -1) {
            cppp.keywords.splice(index, 1);
        }
    }

    function removeKeywords(keywords) {
        for (var i = 0; i < keywords.length; ++i) {
            removeKeyword(keywords[i]);
        }
    }

    function addKeywords(keywords) {
        // (Ruben) Done one by one as if you just push them all, Monaco complains that they're not strings, but as
        // far as I can tell, they indeed are all strings. This somehow fixes it. If you know how to fix it, plz go
        for (var i = 0; i < keywords.length; ++i) {
            cppp.keywords.push(keywords[i]);
        }
    }

    // We remove everything that's not an identifier, underscore reserved name and not an official C++ keyword...
    // Regarding #617, final is a identifier with special meaning, not a fully qualified keyword
    removeKeywords(['abstract', 'amp', 'array', 'cpu', 'delegate', 'each', 'event', 'finally', 'gcnew',
        'generic', 'in', 'initonly', 'interface', 'interior_ptr', 'internal', 'literal', 'partial', 'pascal',
        'pin_ptr', 'property', 'ref', 'restrict', 'safe_cast', 'sealed', 'title_static', 'where']);

    addKeywords(['alignas', 'alignof', 'and', 'and_eq', 'asm', 'bitand', 'bitor', 'char8_t', 'char16_t',
        'char32_t', 'compl', 'concept', 'consteval', 'constinit', 'co_await', 'co_return', 'co_yield', 'not', 'not_eq',
        'or', 'or_eq', 'requires', 'xor', 'xor_eq']);

    return cppp;
}

var def = definition();

monaco.languages.register({id: 'cppp'});
monaco.languages.setLanguageConfiguration('cppp', cpp.conf);
monaco.languages.setMonarchTokensProvider('cppp', def);

export = def;
