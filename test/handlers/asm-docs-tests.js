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

import { expect } from 'chai';
import express from 'express';

import { withAssemblyDocumentationProviders } from '../../lib/handlers/assembly-documentation';
import { chai } from '../utils';

/** Test matrix of architecture to [opcode, tooptip, html, url] */
export const TEST_MATRIX = {
    6502: [['lda', 'Load Accumulator with Memory', 'data is transferred from memory to the accumulator', 'https://www.pagetable.com/c64ref/6502/']],
    amd64: [
        ['mov', 'Copies the second operand', 'Copies the second operand', 'www.felixcloutier.com'],
        ['shr', 'Shifts the bits in the first operand', 'Shifts the bits in the first operand', 'www.felixcloutier.com'], // regression test for #3541
    ],
    arm32: [['mov', 'writes an immediate value', 'writes an immediate value to the destination register', 'https://developer.arm.com/documentation/']],
    avr: [['mov', 'Copy Register', 'makes a copy of one register into another', 'https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf']],
    java: [['iload_0', 'Load int from local variable', 'Load int from local variable', 'https://docs.oracle.com/javase/specs/jvms/se16/html/']],
};

describe('Assembly Documentation API', () => {
    let app;

    before(() => {
        app = express();
        const router = express.Router();
        withAssemblyDocumentationProviders(router);
        app.use('/api', router);
    });

    it('should return 404 for unknown architecture', async () => {
        const res = await chai.request(app).get(`/api/asm/not_an_arch/mov`)
            .set('Accept', 'application/json');
        expect(res).to.have.status(404);
        expect(res).to.be.json;
        expect(res.body).to.deep.equal({ error: `No documentation for 'not_an_arch'` });
    });

    for (const [arch, cases] of Object.entries(TEST_MATRIX)) {
        for (const [opcode, tooltip, html, url] of cases) {
            it(`should process ${arch} text requests`, async () => {
                const res = await chai.request(app).get(`/api/asm/${arch}/${opcode}`)
                    .set('Accept', 'text/plain');
                expect(res).to.have.status(200);
                expect(res).to.be.html;
                expect(res.text).to.contain(html);
            });
    
            it(`should process ${arch} json requests`, async () => {
               const res = await chai.request(app).get(`/api/asm/${arch}/${opcode}`)
                   .set('Accept', 'application/json');
    
               expect(res).to.have.status(200);
               expect(res).to.be.json;
               expect(res.body.html).to.contain(html);
               expect(res.body.tooltip).to.contain(tooltip);
               expect(res.body.url).to.contain(url);
            });
    
            it(`should return 404 for ${arch} unknown opcode requests`, async () => {
                const res = await chai.request(app).get(`/api/asm/${arch}/not_an_opcode`)
                    .set('Accept', 'application/json');
                expect(res).to.have.status(404);
                expect(res).to.be.json;
                expect(res.body).to.deep.equal({ error: 'Unknown opcode \'NOT_AN_OPCODE\'' });
            });
    
            it(`should return 406 for ${arch} bad accept type requests`, async () => {
               const res = await chai.request(app).get(`/api/asm/${arch}/${opcode}`)
                   .set('Accept', 'application/pdf');
               expect(res).to.have.status(406);
            });
        }
    }
});
