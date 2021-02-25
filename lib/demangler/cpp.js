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

import { BaseDemangler } from './base';

const LabelMetadata = [
    {ident: new RegExp('C1E[a-zA-Z0-9_$]*$'), description: 'complete object constructor'},
    {ident: new RegExp('C2E[a-zA-Z0-9_$]*$'), description: 'base object constructor'},
    {ident: new RegExp('C3E[a-zA-Z0-9_$]*$'), description: 'complete object allocating constructor'},
    {ident: new RegExp('D0Ev$'), description: 'deleting destructor'},
    {ident: new RegExp('D1Ev$'), description: 'complete object destructor'},
    {ident: new RegExp('D2Ev$'), description: 'base object destructor'},
];

export class CppDemangler extends BaseDemangler {
    static get key() { return 'cpp'; }

    getMetadata(symbol) {
        return LabelMetadata.filter(metadata => metadata.ident.test(symbol));
    }
}
