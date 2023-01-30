// Copyright (c) 2017, Compiler Explorer Authors
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
import mockfs from 'mock-fs';

import {CompilationQueue} from '../../lib/compilation-queue';
import {HealthCheckHandler} from '../../lib/handlers/health-check';
import {chai} from '../utils';

describe('Health checks', () => {
    let app;
    let compilationQueue;

    let handler;

    beforeEach(() => {
        compilationQueue = new CompilationQueue(1);
        app = express();
        handler = new HealthCheckHandler(compilationQueue);
        app.use('/hc', handler.handle);
    });

    afterEach(() => {
        handler.internal_clear_interval();
    });

    it('should respond with OK', async () => {
        const res = await chai.request(app).get('/hc');
        res.should.have.status(200);
        res.text.should.be.eql('Everything is awesome');
    });
});

describe('Health checks on disk', () => {
    let app;

    let handler;
    let handler2;

    before(() => {
        const compilationQueue = new CompilationQueue(1);

        app = express();
        handler = new HealthCheckHandler(compilationQueue, '/fake/.nonexist');
        app.use('/hc', handler.handle);
        handler2 = new HealthCheckHandler(compilationQueue, '/fake/.health');
        app.use('/hc2', handler2.handle);

        mockfs({
            '/fake': {
                '.health': 'Everything is fine',
            },
        });
    });

    after(() => {
        mockfs.restore();
        handler.internal_clear_interval();
        handler2.internal_clear_interval();
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
