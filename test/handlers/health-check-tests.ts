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
import {HealthcheckController} from '../../lib/handlers/api/healthcheck-controller.js';

describe('Health checks', () => {
    let app: express.Express;
    let compilationQueue: CompilationQueue;

    beforeEach(() => {
        const compileHandlerMock = {
            hasLanguages: () => true,
        };
        compilationQueue = new CompilationQueue(1, 0, 0);
        app = express();
        const controller = new HealthcheckController(compilationQueue, null, compileHandlerMock, false);
        app.use(controller.createRouter());
    });

    it('should respond with OK', async () => {
        await request(app).get('/healthcheck').expect(200, 'Everything is awesome');
    });

    it('should use compilation queue', async () => {
        let count = 0;
        // @ts-expect-error: bypass the private _queue property
        compilationQueue._queue.on('active', () => {
            count++;
        });
        await request(app).get('/healthcheck');
        expect(count).toEqual(1);
    });
});

describe('Health checks without lang/comp', () => {
    let app: express.Express;
    let compilationQueue: CompilationQueue;

    beforeEach(() => {
        const compileHandlerMock = {
            hasLanguages: () => false,
        };
        compilationQueue = new CompilationQueue(1, 0, 0);
        app = express();
        const controller = new HealthcheckController(compilationQueue, null, compileHandlerMock, false);
        app.use(controller.createRouter());
    });

    it('should respond with error', async () => {
        await request(app).get('/healthcheck').expect(500);
    });
});

describe('Health checks without lang/comp but in execution worker mode', () => {
    let app: express.Express;
    let compilationQueue: CompilationQueue;

    beforeEach(() => {
        const compileHandlerMock = {
            hasLanguages: () => false,
        };
        compilationQueue = new CompilationQueue(1, 0, 0);
        app = express();
        const controller = new HealthcheckController(compilationQueue, null, compileHandlerMock, true);
        app.use(controller.createRouter());
    });

    it('should respond with ok', async () => {
        await request(app).get('/healthcheck').expect(200);
    });
});

describe('Health checks on disk', () => {
    let healthyApp: express.Express;
    let unhealthyApp: express.Express;

    beforeAll(() => {
        const compileHandlerMock = {
            hasLanguages: () => true,
        };
        const compilationQueue = new CompilationQueue(1, 0, 0);

        healthyApp = express();
        const hc1 = new HealthcheckController(compilationQueue, '/fake/.health', compileHandlerMock, false);
        healthyApp.use(hc1.createRouter());

        unhealthyApp = express();
        const hc2 = new HealthcheckController(compilationQueue, '/fake/.nonexist', compileHandlerMock, false);
        unhealthyApp.use(hc2.createRouter());

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
        await request(unhealthyApp).get('/healthcheck').expect(500);
    });

    it('should respond with OK and file contents when found', async () => {
        await request(healthyApp)
            .get('/healthcheck')
            .expect(200, 'Everything is fine')
            .expect('content-type', /text\/html/);
    });
});
