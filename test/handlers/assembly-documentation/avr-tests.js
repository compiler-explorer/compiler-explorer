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

import { AVRDocumentationHandler } from '../../../lib/handlers/assembly-documentation/avr';
import { chai } from '../../utils';

describe('AVR assembly documentation', () => {
    let app;
    before(() => {
        app = express();
        const handler = new AVRDocumentationHandler();
        app.use('/asm/:opcode', handler.handle.bind(handler));
    });

    it('returns 404 for unknown opcodes', () => {
        return chai.request(app).get('/asm/mov_oh_wait')
            .then(res => {
                res.should.have.status(404);
                res.should.be.json;
                res.body.should.deep.equal({ error: 'Unknown opcode \'MOV_OH_WAIT\'' });
            }).catch(e => { throw e; });
    });

    it('responds to accept=text requests', () => {
        return chai.request(app).get('/asm/mov')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.contain('makes a copy of one register into another');
            }).catch(e => { throw e; });
    });

    it('responds to accept=json requests', () => {
        return chai.request(app).get('/asm/mov')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.html.should.contain('makes a copy of one register into another');
                res.body.tooltip.should.contain('Copy Register');
                res.body.url.should.contain('https://ww1.microchip.com/downloads/en/DeviceDoc/AVR-InstructionSet-Manual-DS40002198.pdf');
            }).catch(e => { throw e; });
    });

    it('should return 406 on bad accept type', () => {
        return chai.request(app).get('/asm/mov')
            .set('Accept', 'application/pdf')
            .then(res => {
                res.should.have.status(406);
            }).catch(e => { throw e; });
    });
});
