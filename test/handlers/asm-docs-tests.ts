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

import express from 'express';
import request from 'supertest';
import {beforeAll, describe, expect, it} from 'vitest';

import {AssemblyDocumentationController} from '../../lib/handlers/api/assembly-documentation-controller.js';

/** Test matrix of architecture to [opcode, tooptip, html, url] */
export const TEST_MATRIX: Record<PropertyKey, [string, string, string, string][]> = {
    6502: [
        [
            'lda',
            'Load Accumulator with Memory',
            'data is transferred from memory to the accumulator',
            'https://www.pagetable.com/c64ref/6502/',
        ],
    ],
    amd64: [
        ['mov', 'Copies the second operand', 'Copies the second operand', 'www.felixcloutier.com'],
        [
            'shr',
            'Shifts the bits in the first operand',
            'Shifts the bits in the first operand',
            'www.felixcloutier.com',
        ], // regression test for #3541
    ],
    arm32: [
        [
            'mov',
            'writes an immediate value',
            'writes an immediate value to the destination register',
            'https://developer.arm.com/documentation/',
        ],
    ],
    aarch64: [
        [
            'mov',
            'Read active elements from the source predicate',
            '<p>Read active elements from the source predicate',
            'https://developer.arm.com/documentation/',
        ],
    ],
    avr: [
        [
            'mov',
            'Copy Register',
            'makes a copy of one register into another',
            'https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf',
        ],
    ],
    java: [
        [
            'iload_0',
            'Load int from local variable',
            'Load int from local variable',
            'https://docs.oracle.com/javase/specs/jvms/se18/html/',
        ],
    ],
    llvm: [
        [
            'ret',
            'There are two forms of the ‘ret’ instruction',
            '<span id="i-ret"></span><h4>',
            'https://llvm.org/docs/LangRef.html#ret-instruction',
        ],
    ],
    ptx: [
        [
            'add',
            'Add two values.',
            '<p>Performs addition and writes the resulting value into a destination register.</p>',
            '',
        ],
    ],
    powerpc: [
        [
            'addc',
            'Add Carrying',
            '<p>The <strong>addc</strong> and <strong>a</strong> instructions place the sum of the contents of general-purpose register (GPR) <em>RA</em> and GPR <em>RB</em> into the target GPR <em>RT</em>.</p>',
            'https://www.ibm.com/docs/en/aix/7.3?topic=set-addc-add-carrying-instruction',
        ],
    ],
    sass: [['FADD', 'FP32 Add', 'FP32 Add', 'https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14']],
};

describe('Assembly Documentation API', () => {
    let app: express.Express;

    beforeAll(() => {
        app = express();
        const controller = new AssemblyDocumentationController();
        app.use('/', controller.createRouter());
    });

    it('should return 404 for unknown architecture', async () => {
        await request(app)
            .get(`/api/asm/not_an_arch/mov`)
            .set('Accept', 'application/json')
            .expect('Content-Type', /json/)
            .expect(404, {error: `No documentation for 'not_an_arch'`});
    });

    for (const [arch, cases] of Object.entries(TEST_MATRIX)) {
        for (const [opcode, tooltip, html, url] of cases) {
            it(`should process ${arch} text requests`, async () => {
                const res = await request(app)
                    .get(`/api/asm/${arch}/${opcode}`)
                    .set('Accept', 'text/plain')
                    .expect('Content-Type', /html/)
                    .expect(200);
                expect(res.text).toContain(html);
            });
            it(`should process ${arch} json requests`, async () => {
                const res = await request(app)
                    .get(`/api/asm/${arch}/${opcode}`)
                    .set('Accept', 'application/json')
                    .expect('Content-Type', /json/)
                    .expect(200);
                expect(res.body.html).toContain(html);
                expect(res.body.tooltip).toContain(tooltip);
                expect(res.body.url).toContain(url);
            });

            it(`should return 404 for ${arch} unknown opcode requests`, async () => {
                await request(app)
                    .get(`/api/asm/${arch}/not_an_opcode`)
                    .set('Accept', 'application/json')
                    .expect('Content-Type', /json/)
                    .expect(404, {error: "Unknown opcode 'NOT_AN_OPCODE'"});
            });

            it(`should return 406 for ${arch} bad accept type requests`, async () => {
                await request(app).get(`/api/asm/${arch}/${opcode}`).set('Accept', 'application/pdf').expect(406);
            });
        }
    }
});
