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

    describe('compare and carry-less multiply pseudo-ops', () => {
        it.each([
            ['cmpeqps', 'cmpps'],
            ['cmpltpd', 'cmppd'],
            ['cmpneqss', 'cmpss'],
            ['cmpunordps', 'cmpps'],
            ['CMPNLESS', 'cmpss'],
            ['vcmpeqps', 'vcmpps'],
            ['vcmpltpd', 'vcmppd'],
            ['vcmpordss', 'vcmpss'],
            ['vcmpeq_uqpd', 'vcmppd'],
            ['vcmplt_osps', 'vcmpps'],
            ['vcmptrue_usss', 'vcmpss'],
            ['vcmpfalse_oqps', 'vcmpps'],
            ['vpcmpltb', 'vpcmpb'],
            ['vpcmpnled', 'vpcmpd'],
            ['vpcmpneqq', 'vpcmpq'],
            ['vpcmpnltw', 'vpcmpw'],
            ['vpcmpequb', 'vpcmpub'],
            ['vpcmpltuw', 'vpcmpuw'],
            ['VPCMPNLEUD', 'vpcmpud'],
            ['vpcmpnequq', 'vpcmpuq'],
            ['pclmullqlqdq', 'pclmulqdq'],
            ['pclmulhqlqdq', 'pclmulqdq'],
            ['vpclmulhqhqdq', 'vpclmulqdq'],
        ])('should document %s with the page of %s', (pseudoOp, instruction) => {
            expect(urlFor(instruction)).toBeDefined();
            expect(urlFor(pseudoOp)).toBe(urlFor(instruction));
        });

        it('should fall back to the legacy page when the VEX form has none', () => {
            expect(urlFor('vcmpeqsd')).toBeDefined();
            expect(urlFor('vcmpeqsd')).toBe(urlFor('cmpsd'));
        });

        it('should keep the instructions that look like pseudo-ops on their own pages', () => {
            expect(urlFor('vpcmpeqb')).toContain('PCMPEQB');
            expect(urlFor('vpcmpgtb')).toContain('PCMPGTB');
        });

        it('should not invent pseudo-ops', () => {
            for (const opcode of ['cmpfooPS', 'vcmpeqxx', 'vpcmpgtub', 'vpcmpltx', 'pclmulxqlqdq', 'cmpeq_uqps']) {
                expect(urlFor(opcode)).toBeUndefined();
            }
        });
    });
});
