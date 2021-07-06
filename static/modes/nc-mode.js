// Copyright (c) 2018, Compiler Explorer Authors
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

// We need to ensure we use proper keywords for the Monaco Editor matcher. Note how
// https://github.com/Microsoft/monaco-languages/ lacks, as far as I can tell, proper C support. We cheat and use C++
function definition() {
    var nc = $.extend(true, {}, cpp.language); // deep copy
    // https://en.cppreference.com/w/c/keyword
    nc.keywords = ['auto', 'break', 'case', 'char', 'const', 'continue', 'default',
        'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'inline',
        'int', 'long', 'register', 'restrict', 'return', 'short', 'signed', 'sizeof', 'static',
        'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while',
        '_Alignas', '_Alignof', '_Atomic', '_Bool', '_Complex', '_Generic', '_Imaginary',
        '_Noreturn', '_Static_assert', '_Thread_local',
    ];
    return nc;
}

var def = definition();

monaco.languages.register({id: 'nc'});
monaco.languages.setLanguageConfiguration('nc', cpp.conf);
monaco.languages.setMonarchTokensProvider('nc', def);

module.exports = def;
