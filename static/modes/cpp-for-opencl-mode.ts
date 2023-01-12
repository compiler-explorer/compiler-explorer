// Copyright (c) 2018, 2021, Arm Ltd & Compiler Explorer Authors
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

// We need to create a new definition for C++ for OpenCL so we can remove invalid keywords

function definition(): monaco.languages.IMonarchLanguage {
    const cppForOpenCL = $.extend(true, {}, cppp); // deep copy

    function addKeywords(keywords) {
        // (Ruben) Done one by one as if you just push them all, Monaco complains that they're not strings, but as
        // far as I can tell, they indeed are all strings. This somehow fixes it. If you know how to fix it, plz go
        for (let i = 0; i < keywords.length; ++i) {
            cppForOpenCL.keywords.push(keywords[i]);
        }
    }

    function vectorTypes(basename) {
        return [basename + '2', basename + '3', basename + '4', basename + '8', basename + '16'];
    }
    // Keywords for C++ for OpenCL
    addKeywords([
        '__global',
        'global',
        '__local',
        'local',
        '__constant',
        'constant',
        '__private',
        'private',
        '__generic',
        'generic',
        '__kernel',
        'kernel',
        'uniform',
        'pipe',
        '__read_only',
        'read_only',
        '__write_only',
        'write_only',
        '__read_write',
        'read_write',
        'bool',
        'uchar',
        'ushort',
        'uint',
        'ulong',
        'cl_mem_fence_flags',
        'event_t',
        'reserve_id_t',
        'ndrange_t',
        'queue_t',
        'image2d_t',
        'image3d_t',
        'image2d_array_t',
        'image1d_t',
        'image1d_array_t',
        'image2d_depth_t',
        'image1d_buffer_t',
        'image2d_array_depth_t',
        'sampler_t',
        'uintptr_t',
        'intptr_t',
        'ptrdiff_t',
        'size_t',
    ]);
    addKeywords(vectorTypes('char'));
    addKeywords(vectorTypes('short'));
    addKeywords(vectorTypes('int'));
    addKeywords(vectorTypes('long'));
    addKeywords(vectorTypes('uchar'));
    addKeywords(vectorTypes('ushort'));
    addKeywords(vectorTypes('uint'));
    addKeywords(vectorTypes('ulong'));
    addKeywords(vectorTypes('half'));
    addKeywords(vectorTypes('float'));
    addKeywords(vectorTypes('double'));

    cppForOpenCL.floatsuffix = /[fFhH]?/;

    return cppForOpenCL;
}

monaco.languages.register({id: 'cpp-for-opencl'});
monaco.languages.setLanguageConfiguration('cpp-for-opencl', cpp.conf);
monaco.languages.setMonarchTokensProvider('cpp-for-opencl', definition());

export {};
