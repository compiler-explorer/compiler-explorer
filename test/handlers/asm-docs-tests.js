// Copyright (c) 2017, Matt Godbolt
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

const chai = require('chai'),
    asmDocsApi = require('../../lib/handlers/asm-docs-api'),
    express = require('express');

chai.use(require('chai-http'));
chai.should();

describe('Assembly documents', () => {
    const app = express();
    const handler = new asmDocsApi.Handler();
    app.use('/asm/:opcode', handler.handle.bind(handler));

    // We don't serve a 404 for unknown opcodes as it allows the not-an-opcode to be cached.
    it('should respond with "unknown opcode" for unknown opcodes', () => {
        return chai.request(app)
            .get('/asm/NOTANOPCODE')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.equal('Unknown opcode');
            })
            .catch(function (err) {
                throw err;
            });
    });

    it('should respond to text requests', () => {
        return chai.request(app)
            .get('/asm/mov')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.contain('Copies the second operand');
            })
            .catch(function (err) {
                throw err;
            });
    });

    it('should respond to json requests', () => {
        return chai.request(app)
            .get('/asm/mov')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.found.should.equal(true);
                res.body.result.html.should.contain('Copies the second operand');
                res.body.result.tooltip.should.contain('Copies the second operand');
                res.body.result.url.should.contain('www.felixcloutier.com');
            })
            .catch(function (err) {
                throw err;
            });
    });
    it('should respond to json for unknown opcodes', () => {
        return chai.request(app)
            .get('/asm/NOANOPCODE')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.found.should.equal(false);
            })
            .catch(function (err) {
                throw err;
            });
    });

    it('should handle at&t syntax', () => {
        return chai.request(app)
            .get('/asm/addq')
            .then(res => {
                res.should.have.status(200);
                res.should.be.html;
                res.text.should.contain('Adds the destination operand');
            })
            .catch(function (err) {
                throw err;
            });
    });
});
