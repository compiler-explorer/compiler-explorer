// Copyright (c) 2024, Compiler Explorer Authors
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
import request from 'supertest';
import {describe, it} from 'vitest';

import {cached, cors} from '../../lib/handlers/middleware.js';

describe('middleware functions', () => {
    it('adds cache controls headers with cached()', async () => {
        const app = express();
        app.use('/', cached, (_, res) => res.send('Hello World!'));
        await request(app)
            .get('/')
            .expect(200, 'Hello World!')
            .expect('cache-control', /public, max-age=\d+, must-revalidate/);
    });

    it('adds cors headers with cors()', async () => {
        const app = express();
        app.use('/', cors, (_, res) => res.send('Hello World!'));
        await request(app)
            .get('/')
            .expect(200, 'Hello World!')
            .expect('access-control-allow-origin', '*')
            .expect('access-control-allow-headers', 'Origin, X-Requested-With, Content-Type, Accept');
    });
});
