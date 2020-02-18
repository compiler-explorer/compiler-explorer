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
    healthCheck = require('../../lib/handlers/health-check'),
    express = require('express'),
    mockfs = require('mock-fs');

chai.use(require("chai-http"));
chai.should();

describe('Health checks', () => {
    const app = express();
    app.use('/hc', new healthCheck.HealthCheckHandler().handle);

    it('should respond with OK', async () => {
        const res = await chai.request(app).get('/hc');
        res.should.have.status(200);
        res.text.should.be.eql('Everything is awesome');
    });
});

describe('Health checks on disk', () => {
    const app = express();
    app.use('/hc', new healthCheck.HealthCheckHandler('/fake/.nonexist').handle);
    app.use('/hc2', new healthCheck.HealthCheckHandler('/fake/.health').handle);

    before(() => {
        mockfs({
            '/fake': {
                '.health': 'Everything is fine'
            }
        });
    });

    after(() => {
        mockfs.restore();
    });

    it('should respond with 500 when file not found', async () => {
        const res = await chai.request(app).get('/hc');
        res.should.have.status(500);
    });


    it('should respond with OK and file contents when found', async () => {
        const res = await chai.request(app).get('/hc2');
        res.should.have.status(200);
        res.text.should.be.eql('Everything is fine');
    });
});
