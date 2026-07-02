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
import * as cpp from 'monaco-editor/esm/vs/basic-languages/cpp/cpp';

function definition(): monaco.languages.IMonarchLanguage {
    const co2 = $.extend(true, {}, cpp.language);

    co2.keywords = [
        'alignas',
        'alignof',
        'auto',
        'bool',
        'break',
        'case',
        'char',
        'const',
        'constexpr',
        'continue',
        'default',
        'do',
        'double',
        'else',
        'enum',
        'extern',
        'false',
        'float',
        'fn',
        'for',
        'goto',
        'if',
        'inline',
        'int',
        'long',
        'mod',
        'nullptr',
        'pub',
        'register',
        'restrict',
        'return',
        'short',
        'signed',
        'sizeof',
        'static',
        'static_assert',
        'struct',
        'switch',
        'thread_local',
        'true',
        'type',
        'typedef',
        'typeof',
        'typeof_unqual',
        'union',
        'unsafe',
        'unsigned',
        'use',
        'void',
        'volatile',
        'while',
        '_Alignas',
        '_Alignof',
        '_Atomic',
        '_BitInt',
        '_Bool',
        '_Complex',
        '_Decimal128',
        '_Decimal32',
        '_Decimal64',
        '_Generic',
        '_Imaginary',
        '_Noreturn',
        '_Pragma',
        '_Static_assert',
        '_Thread_local',
    ];

    // Add CO2-specific patterns before original tokenizer rules
    const origToken = co2.tokenizer.root;
    co2.tokenizer.root = [
        // CO2 attributes
        [/^#!\[.*?\]/, 'annotation'],
        [/#\[.*?\]/, 'annotation'],

        // CO2 paths: foo::bar::baz
        [/[a-zA-Z_]\w*(?:::[a-zA-Z_]\w*)+/, 'type.identifier'],

        ...origToken,
    ];

    return co2;
}

const def = definition();

monaco.languages.register({id: 'co2'});
monaco.languages.setLanguageConfiguration('co2', cpp.conf);
monaco.languages.setMonarchTokensProvider('co2', def);

export default def;
