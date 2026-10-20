// Copyright (c) 2025, Compiler Explorer Authors
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

export const UrlTestCases = [
    {
        remoteUrl: 'https://example.com/path/to/resource',
        language: 'cpp',
        expectedId: 'example_com_path_to_resource_cpp',
    },
    {
        remoteUrl: 'https://example.com',
        language: 'java',
        expectedId: 'example_com__java',
    },
    {
        remoteUrl: 'https://sub.domain.com/a/b/c/',
        language: 'rust',
        expectedId: 'sub_domain_com_a_b_c__rust',
    },
    {
        remoteUrl: 'https://godbolt.org:443/gpu',
        language: 'c++',
        expectedId: 'godbolt_org_gpu_c++',
    },
    {
        remoteUrl: 'https://godbolt.org:443/winprod',
        language: 'c++',
        expectedId: 'godbolt_org_winprod_c++',
    },
    {
        remoteUrl: 'https://godbolt.org:443',
        language: 'c++',
        expectedId: 'godbolt_org__c++',
    },
];
