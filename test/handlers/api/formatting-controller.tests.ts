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
import {beforeAll, describe, it} from 'vitest';

import {FormattingService} from '../../../lib/formatting-service.js';
import {FormattingController} from '../../../lib/handlers/api/formatting-controller.js';
import {fakeProps} from '../../../lib/properties.js';

describe('FormattingController', () => {
    let app;

    beforeAll(async () => {
        app = express();
        const formattingService = new FormattingService();
        const formattingController = new FormattingController(formattingService);
        await formattingService.initialize(
            fakeProps({
                formatters: 'formatt:badformatt',
                'formatter.formatt.exe': process.platform === 'win32' ? 'cmd' : 'echo',
                'formatter.formatt.type': 'clangformat',
                'formatter.formatt.version': 'Release',
                'formatter.formatt.name': 'FormatT',
            }),
        );
        app.use(express.json());
        app.use(formattingController.createRouter());
    });

    it('should not go through with invalid tools', async () => {
        await request(app)
            .post('/api/format/invalid')
            .set('Accept', 'application/json')
            .set('Content-Type', 'application/json')
            .expect('Content-Type', /json/)
            .expect(422, {exit: 2, answer: "Unknown format tool 'invalid'"});
    });

    it('should reject requests with no content type', async () => {
        await request(app)
            .post('/api/format/formatt')
            .send('{ base: "something", source: "int main() {}" }')
            .expect('Content-Type', /json/)
            .expect(400);
    });

    it('should not go through with invalid base styles', async () => {
        await request(app)
            .post('/api/format/formatt')
            .send({
                base: 'bad-base',
                source: 'i am source',
            })
            .set('Accept', 'application/json')
            .set('Content-Type', 'application/json')
            .expect(422, {exit: 3, answer: "Style 'bad-base' is not supported"})
            .expect('Content-Type', /json/);
    });

    it('should reject requests with no source', async () => {
        await request(app)
            .post('/api/format/formatt')
            .send({
                base: 'bad-base',
            })
            .set('Accept', 'application/json')
            .expect(400, {exit: 0, answer: ''})
            .expect('Content-Type', /json/);
    });
});
