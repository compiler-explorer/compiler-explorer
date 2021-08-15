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

import { ClangCompiler } from '../../lib/compilers';
import { chai, makeCompilationEnvironment } from '../utils';

const expect = chai.expect;

describe('clang tests', () => {
    const languages = {'c++': {id: 'c++'}};

    const info = {
        exe: null,
        remote: true,
        lang: 'c++',
        ldPath: [],
    };

    describe('device code...', async () => {
        const clang = new ClangCompiler(info, makeCompilationEnvironment({languages}));
        it('Should return null for non-device code', () => {
            expect(clang.splitDeviceCode('')).to.be.null;
            expect(clang.splitDeviceCode('mov eax, 00h\nadd r0, r0, #1\n')).to.be.null;
        });
        it('should separate out bundles ', () => {
            expect(clang.splitDeviceCode(`# __CLANG_OFFLOAD_BUNDLE____START__ openmp-x86_64-unknown-linux-gnu
    i am some
    linux remote stuff
# __CLANG_OFFLOAD_BUNDLE____END__ openmp-x86_64-unknown-linux-gnu

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
    whereas
    i am host code
# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
`)).to.deep.equal({
                'host-x86_64-unknown-linux-gnu': '    whereas\n    i am host code\n',
                'openmp-x86_64-unknown-linux-gnu': '    i am some\n    linux remote stuff\n',
            });
        });
    });
});
