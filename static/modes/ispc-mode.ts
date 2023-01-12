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

import $ from 'jquery';

import * as monaco from 'monaco-editor';
import * as cpp from 'monaco-editor/esm/vs/basic-languages/cpp/cpp';

function definition(): monaco.languages.IMonarchLanguage {
    const ispc = $.extend(true, {}, cpp.language); // deep copy

    ispc.tokenPostfix = '.ispc';

    ispc.keywords.push(
        'cbreak',
        'ccontinue',
        'cdo',
        'cfor',
        'cif',
        'creturn',
        'cwhile',
        'delete',
        'export',
        'foreach',
        'foreach_active',
        'foreach_tiled',
        'foreach_unique',
        'int16',
        'int32',
        'int64',
        'int8',
        'launch',
        'new',
        'operator',
        'programCount',
        'programIndex',
        'reference',
        'size_t',
        'soa',
        'sync',
        'task',
        'taskCount',
        'taskCount0',
        'taskCount1',
        'taskCount2',
        'taskIndex',
        'taskIndex0',
        'taskIndex1',
        'taskIndex2',
        'threadCount',
        'threadIndex',
        'uint16',
        'uint32',
        'uint64',
        'uint8',
        'uniform',
        'unmasked',
        'varying'
    );
    return ispc;
}

monaco.languages.register({id: 'ispc'});
monaco.languages.setLanguageConfiguration('ispc', cpp.conf);
monaco.languages.setMonarchTokensProvider('ispc', definition());

export {};
