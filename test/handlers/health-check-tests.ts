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
import request from 'supertest';
import {afterAll, beforeAll, beforeEach, describe, expect, it} from 'vitest';

import {CompilationQueue} from '../../lib/compilation-queue.js';
import {HealthCheckHandler} from '../../lib/handlers/health-check.js';

describe('Health checks', () => {
    let app;
    let compilationQueue;

    beforeEach(() => {
        const compileHandlerMock = {
            hasLanguages: () => true,
        };
        compilationQueue = new CompilationQueue(1, 0, 0);
        app = express();
        app.use('/hc', new HealthCheckHandler(compilationQueue, '', compileHandlerMock).handle);
    });

    it('should respond with OK', async () => {
        await request(app).get('/hc').expect(200, 'Everything is awesome');
    });

    it('should use compilation queue', async () => {
        let count = 0;
        compilationQueue._queue.on('active', () => {
            count++;
        });
        await request(app).get('/hc');
        expect(count).toEqual(1);
    });
});

describe('Health checks without lang/comp', () => {
    let app;
    let compilationQueue;

    beforeEach(() => {
        const compileHandlerMock = {
            hasLanguages: () => false,
        };
        compilationQueue = new CompilationQueue(1, 0, 0);
        app = express();
        app.use('/hc', new HealthCheckHandler(compilationQueue, '', compileHandlerMock).handle);
    });

    it('should respond with error', async () => {
        await request(app).get('/hc').expect(500);
    });
});

describe('Health checks on disk', () => {
    let app;

    beforeAll(() => {
        const compileHandlerMock = {
            hasLanguages: () => true,
        };
        const compilationQueue = new CompilationQueue(1, 0, 0);

        app = express();
        app.use('/hc', new HealthCheckHandler(compilationQueue, '/fake/.nonexist', compileHandlerMock).handle);
        app.use('/hc2', new HealthCheckHandler(compilationQueue, '/fake/.health', compileHandlerMock).handle);

        mockfs({
            '/fake': {
                '.health': 'Everything is fine',
            },
        });
    });

    afterAll(() => {
        mockfs.restore();
    });

    it('should respond with 500 when file not found', async () => {
        await request(app).get('/hc').expect(500);
    });

    it('should respond with OK and file contents when found', async () => {
        await request(app).get('/hc2').expect(200, 'Everything is fine');
    });
});
