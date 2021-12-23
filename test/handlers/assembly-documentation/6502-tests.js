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

import { Mos6502DocumentationHandler } from '../../../lib/handlers/assembly-documentation/mos6502';
import { chai } from '../../utils';

describe('6502 assembly documentation', () => {
    let app;
    before(() => {
        app = express();
        const handler = new Mos6502DocumentationHandler();
        app.use('/asm/:opcode', handler.handle.bind(handler));
    });

    it('returns 404 for unknown opcodes', () => {
        return chai.request(app).get('/asm/lda_oh_wait')
            .then(res => {
                res.should.have.status(404);
                res.should.be.json;
                res.body.should.deep.equal({ error: 'Unknown opcode \'LDA_OH_WAIT\'' });
            }).catch(e => { throw e; });
    });

    it('responds to accept=text requests', () => {
        return chai.request(app).get('/asm/lda')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.contain('data is transferred from memory to the accumulator');
            }).catch(e => { throw e; });
    });

    it('responds to accept=json requests', () => {
        return chai.request(app).get('/asm/lda')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.html.should.contain('data is transferred from memory to the accumulator');
                res.body.tooltip.should.contain('Load Accumulator with Memory');
                res.body.url.should.contain('https://www.pagetable.com/c64ref/6502/');
            }).catch(e => { throw e; });
    });

    it('should return 406 on bad accept type', () => {
        return chai.request(app).get('/asm/lda')
            .set('Accept', 'application/pdf')
            .then(res => {
                res.should.have.status(406);
            }).catch(e => { throw e; });
    });
});
