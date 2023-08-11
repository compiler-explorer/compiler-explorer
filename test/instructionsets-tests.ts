// Copyright (c) 2022, Compiler Explorer Authors
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

import {InstructionSets} from '../lib/instructionsets.js';

describe('InstructionSets', async () => {
    it('should recognize aarch64 for clang target', async () => {
        const isets = new InstructionSets();

        return isets
            .getCompilerInstructionSetHint('aarch64-linux-gnu', '/opt/compiler-explorer/clang-11.0.1/bin/clang++')
            .should.eventually.equal('aarch64');
    });

    it('should recognize gcc aarch64 from filepath', async () => {
        const isets = new InstructionSets();

        return isets
            .getCompilerInstructionSetHint(
                false,
                '/opt/compiler-explorer/arm64/gcc-12.1.0/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-g++',
            )
            .should.eventually.equal('aarch64');
    });

    it('should default to amd64 when not apparant', async () => {
        const isets = new InstructionSets();

        return isets
            .getCompilerInstructionSetHint(false, '/opt/compiler-explorer/gcc-12.2.0/bin/g++')
            .should.eventually.equal('amd64');
    });
});
