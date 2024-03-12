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
import request from 'supertest';
import {describe, expect, it} from 'vitest';

import {SourceHandler} from '../../lib/handlers/source.js';

describe('Sources', () => {
    const app = express();
    const handler = new SourceHandler(
        [
            {
                name: 'moose',
                urlpart: 'moose',
                list: async () => [{file: 'file', lang: 'lang', name: 'name'}],
                load: name => Promise.resolve({file: `File called ${name}`}),
            },
        ],
        res => res.setHeader('Yibble', 'boing'),
    );
    app.use('/source', handler.handle.bind(handler));

    it('should list', async () => {
        const res = await request(app)
            .get('/source/moose/list')
            .expect('Content-Type', /json/)
            .expect(200, [{file: 'file', lang: 'lang', name: 'name'}]);
        expect(res.headers['yibble']).toEqual('boing');
    });
    it('should fetch files', async () => {
        const res = await request(app)
            .get('/source/moose/load/Grunkle')
            .expect('Content-Type', /json/)
            .expect(200, {file: 'File called Grunkle'});
        expect(res.headers['yibble']).toEqual('boing');
    });
});
