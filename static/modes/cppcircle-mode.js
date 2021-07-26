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

'use strict';
var $ = require('jquery');
var monaco = require('monaco-editor');
var cpp = require('monaco-editor/esm/vs/basic-languages/cpp/cpp');
var cppp = require('./cppp-mode');

// circle is c++ with a few extra '@'-prefixed keywords.

function definition() {
    var cppcircle = $.extend(true, {}, cppp); // deep copy

    function addKeywords(keywords) {
        // (Ruben) Done one by one as if you just push them all, Monaco complains that they're not strings, but as
        // far as I can tell, they indeed are all strings. This somehow fixes it. If you know how to fix it, plz go
        for (var i = 0; i < keywords.length; ++i) {
            cppcircle.keywords.push(keywords[i]);
        }
    }

    addKeywords(['@array', '@attribute', '@base_count', '@base_offset', '@base_offsets', '@base_type',
        '@base_type_string', '@base_type_strings', '@base_types', '@base_value', '@base_values', '@codegen',
        '@data', '@decl_string', '@dynamic_type', '@embed', '@emit', '@enum_attribute', '@enum_attributes',
        '@enum_count', '@enum_decl_string', '@enum_decl_strings', '@enum_has_attribute', '@enum_name', '@enum_names',
        '@enum_type', '@enum_type_string', '@enum_type_strings', '@enum_types', '@enum_value', '@enum_values',
        '@expand_params', '@expression', '@files', '@func_decl', '@has_attribute', '@include', '@is_class_template',
        '@macro', '@match', '@mauto', '@member', '@member_count', '@member_decl_string', '@member_decl_strings',
        '@member_default', '@member_has_attribute', '@member_has_default', '@member_name', '@member_names',
        '@member_offset', '@member_offsets', '@member_ptr', '@member_ptrs', '@member_type', '@member_type_string',
        '@member_type_strings', '@member_types', '@member_value', '@member_values', '@meta', '@method_count',
        '@method_name', '@method_params', '@method_type', '@mtype', '@mtypes', '@mvoid', '@op', '@pack_nontype',
        '@pack_type', '@parse_expression', '@puts', '@range', '@sfinae', '@statements', '@static_type', '@string',
        '@tattribute', '@type_enum', '@type_id', '@type_name', '@type_string']);

    // Hack to put `@...` keywords in place
    cppcircle.tokenizer.root.unshift(
        [/@\w+/, {cases: {'@keywords': {token: 'keyword.$0'}}}]
    );

    return cppcircle;
}

monaco.languages.register({ id: 'cppcircle' });
monaco.languages.setLanguageConfiguration('cppcircle', cpp.conf);
monaco.languages.setMonarchTokensProvider('cppcircle', definition());
