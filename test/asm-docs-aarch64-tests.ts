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

import {ArmArch64DocumentationProvider} from '../lib/asm-docs/arm.js';

describe('AArch64 asm docs', () => {
    const provider = new ArmArch64DocumentationProvider();
    const tooltipFor = (opcode: string) => provider.getInstructionInformation(opcode)?.tooltip;

    it.each(['LDP', 'STP', 'LDR', 'STR', 'LDUR', 'STUR'])(
        'should describe %s for general-purpose registers',
        opcode => {
            expect(tooltipFor(opcode)).not.toContain('SIMD&FP');
        },
    );

    it.each(['AND', 'EOR', 'ORR', 'NEG', 'MVN'])('should describe the scalar form of %s', opcode => {
        expect(tooltipFor(opcode)).not.toMatch(/vector|SIMD/i);
    });

    it('should describe MOV as a register move', () => {
        expect(tooltipFor('MOV')).toMatch(/^Move \(register\)/);
    });

    it('should still document instructions that only have SIMD or SVE forms', () => {
        expect(tooltipFor('FMLA')).toBeTruthy();
        expect(tooltipFor('PTRUE')).toBeTruthy();
        expect(tooltipFor('SMULL2')).toBeTruthy();
        expect(tooltipFor('UMULL2')).toBeTruthy();
    });
});
