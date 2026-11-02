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

import cppp from './cppp-mode.js';

// Metal Shading Language is C++14 with address spaces, function qualifiers, [[attributes]] and its own
// vector and matrix types.
function definition(): monaco.languages.IMonarchLanguage {
    const metal = $.extend(true, {}, cppp); // deep copy

    function addKeywords(keywords: string[]) {
        // One by one, as in cuda-mode.ts: pushing them all at once makes Monaco reject them.
        for (let i = 0; i < keywords.length; ++i) {
            metal.keywords.push(keywords[i]);
        }
    }

    metal.tokenPostfix = '.metal';

    addKeywords([
        // Function qualifiers
        'kernel',
        'vertex',
        'fragment',
        'visible',
        'stitchable',
        // Address spaces
        'device',
        'constant',
        'thread',
        'threadgroup',
        'threadgroup_imageblock',
        'ray_data',
        'object_data',
        // Scalar types beyond C++
        'half',
        'bfloat',
        'uchar',
        'ushort',
        'uint',
        'ulong',
    ]);

    // Vector and matrix types: float4, half2, uint3, float4x4, packed_float3, ...
    for (const base of [
        'bool',
        'char',
        'uchar',
        'short',
        'ushort',
        'int',
        'uint',
        'long',
        'ulong',
        'half',
        'bfloat',
        'float',
    ]) {
        for (const n of ['2', '3', '4']) {
            addKeywords([base + n, 'packed_' + base + n]);
        }
    }
    for (const base of ['half', 'float']) {
        for (const c of ['2', '3', '4']) {
            for (const r of ['2', '3', '4']) {
                addKeywords([base + c + 'x' + r]);
            }
        }
    }

    // [[thread_position_in_grid]], [[buffer(0)]], ... read as annotations.
    metal.tokenizer.root.unshift([/\[\[[^\]]*\]\]/, 'annotation']);

    return metal;
}

monaco.languages.register({id: 'metal'});
monaco.languages.setLanguageConfiguration('metal', cpp.conf);
monaco.languages.setMonarchTokensProvider('metal', definition());
