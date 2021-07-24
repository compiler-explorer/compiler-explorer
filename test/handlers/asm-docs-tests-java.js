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

import { AsmDocsHandler } from '../../lib/handlers/asm-docs-api-java';
import { chai } from '../utils';

describe('Assembly documents', () => {
    let app;

    before(() => {
        app = express();
        const handler = new AsmDocsHandler();
        app.use('/asm/:opcode', handler.handle.bind(handler));
    });

    it('should respond with "unknown opcode" for unknown opcodes', () => {
        return chai.request(app).get('/asm/NOTANOPCODE')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.equal('Unknown opcode');
            })
            .catch(err => { throw err; });
    });

    it('should respond to text requests', () => {
        return chai.request(app)
            .get('/asm/aaload')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.contain('Load reference from array');
            })
            .catch(err => { throw err; });
    });

    it('should respond to json requests', () => {
        return chai.request(app)
            .get('/asm/aaload')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.found.should.equal(true);
                res.body.result.html.should.contain('Load reference from array');
                res.body.result.tooltip.should.contain('Load reference from array');
                res.body.result.url.should.contain('https://docs.oracle.com/javase/specs/');
            })
            .catch(err => { throw err; });
    });

    it('should respond to json for unknown opcodes', () => {
        return chai.request(app)
            .get('/asm/NOTANOPCODE')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.found.should.equal(false);
            })
            .catch(err => { throw err; });
    });
});
