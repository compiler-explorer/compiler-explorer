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

import express, { Router } from 'express';

import { setup } from '../../../lib/handlers/assembly-documentation/router';
import { chai } from '../../utils';

describe('Assembly Documentation API', () => {
    let app;

    before(() => {
        app = express();
        /** @type {e.Router} */
        const router = Router();
        setup(router);
        app.use('/api', router);
    });

    it('should accept requests to the api', () => {
        return chai.request(app)
            .get('/api/asm/amd64/mov')
            .set('Accept', 'application/json')
            .then(res => {
                res.should.have.status(200);
                res.should.be.json;
            })
            .catch(err => {
                throw err;
            });
    });

    it('should return opcode not found', () => {
        return chai.request(app).get('/api/asm/amd64/notexistingop')
            .set('Accept', 'application/json')
            .then((res) => {
                res.should.have.status(404);
                res.should.be.json;
                res.body.should.deep.equals({ error: `Unknown opcode 'NOTEXISTINGOP'` });
            }).catch(e => { throw e; });
    });

    it('should return architecture not found', () => {
        return chai.request(app).get('/api/asm/notarch/mov')
            .set('Accept', 'application/json')
            .then((res) => {
                res.should.have.status(404);
                res.should.be.json;
                res.body.should.deep.equals({ error: `No documentation for 'notarch'` });
            }).catch(e => { throw e; });
    });
});
