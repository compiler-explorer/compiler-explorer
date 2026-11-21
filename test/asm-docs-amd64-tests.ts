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

import {describe, expect, it} from 'vitest';

import {Amd64DocumentationProvider} from '../lib/asm-docs/amd64.js';

describe('amd64 asm docs', () => {
    const provider = new Amd64DocumentationProvider();
    const urlFor = (opcode: string) => provider.getInstructionInformation(opcode)?.url;

    describe('AT&T extension mnemonics', () => {
        it.each([
            ['movzbl', 'MOVZX.html'],
            ['movzbw', 'MOVZX.html'],
            ['movzbq', 'MOVZX.html'],
            ['movzwl', 'MOVZX.html'],
            ['movzwq', 'MOVZX.html'],
            ['movsbl', 'MOVSX%3AMOVSXD.html'],
            ['movsbw', 'MOVSX%3AMOVSXD.html'],
            ['movsbq', 'MOVSX%3AMOVSXD.html'],
            ['movswl', 'MOVSX%3AMOVSXD.html'],
            ['movswq', 'MOVSX%3AMOVSXD.html'],
            ['movslq', 'MOVSX%3AMOVSXD.html'],
            ['cbtw', 'CBW%3ACWDE%3ACDQE.html'],
            ['cwtl', 'CBW%3ACWDE%3ACDQE.html'],
            ['cltq', 'CBW%3ACWDE%3ACDQE.html'],
            ['cwtd', 'CWD%3ACDQ%3ACQO.html'],
            ['cltd', 'CWD%3ACDQ%3ACQO.html'],
            ['cqto', 'CWD%3ACDQ%3ACQO.html'],
        ])('should document %s', (opcode, page) => {
            expect(urlFor(opcode)).toBe(`https://www.felixcloutier.com/x86/${page}`);
        });
    });

    it('should still document the string move instructions', () => {
        for (const opcode of ['movsb', 'movsw', 'movsl', 'movsq']) {
            expect(urlFor(opcode)).toContain('MOVS%3AMOVSB');
        }
    });
});
