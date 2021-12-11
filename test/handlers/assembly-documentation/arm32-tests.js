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

import { Arm32DocumentationHandler } from '../../../lib/handlers/assembly-documentation/arm32';
import { chai } from '../../utils';

describe('arm32 assembly documentation', () => {
    let app;
    before(() => {
        app = express();
        const handler = new Arm32DocumentationHandler();
        app.use('/asm/:opcode', handler.handle.bind(handler));
    });

    it('returns 404 for unknown opcodes', () => {
        return chai.request(app).get('/asm/mov_oh_wait')
            .then(res => {
                res.should.have.status(404);
                res.should.be.json;
                res.body.should.deep.equal({ error: 'Unknown opcode' });
            }).catch(e => { throw e; });
    });

    it('responds to accept=text requests', () => {
        return chai.request(app).get('/asm/mov')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.contain('writes an immediate value to the destination register');
            }).catch(e => { throw e; });
    });

    it('responds to accept=json requests', () => {
        return chai.request(app).get('/asm/mov')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.html.should.contain('writes an immediate value');
                res.body.tooltip.should.contain('writes an immediate value');
                res.body.url.should.contain('https://developer.arm.com/documentation/');
            }).catch(e => { throw e; });
    });

    it('should respond to json for unknown opcodes', () => {
        return chai.request(app)
            .get('/asm/NOANOPCODE')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(404);
                res.should.be.json;
            }).catch(e => { throw e; });
    });

    it('should return 406 on bad accept type', () => {
        return chai.request(app).get('/asm/mov')
            .set('Accept', 'application/pdf')
            .then(res => {
                res.should.have.status(406);
            }).catch(e => { throw e; });
    });

    it('should handle opcodes with a conditional type suffix that are not conditionals', () => {
        return chai.request(app)
            .get('/asm/adcs')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.contain('adds an immediate value and the Carry flag');
            }).catch(e => { throw e; });
    });

    it('should handle conditional opcodes', () => {
        return chai.request(app)
            .get('/asm/beq')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.contain('If equal, <p>Branch causes a branch');
            }).catch(e => { throw e; });
    });

    it('should respond with "unknown opcode" for unknown opcodes that can be parsed as conditional', () => {
        return chai.request(app)
            .get('/asm/jne')
            .then(res => {
                res.should.have.status(404);
                res.should.be.json;
                res.body.should.deep.equal({ error: 'Unknown opcode' });
            }).catch(e => { throw e; });
    });
});
