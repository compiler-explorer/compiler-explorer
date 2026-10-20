// Copyright (c) 2023, Compiler Explorer Authors
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
import {beforeAll, describe, expect, it} from 'vitest';

import {SiteTemplateController} from '../../lib/handlers/api/site-template-controller.js';
import {getSiteTemplates} from '../../lib/site-templates.js';

describe('Site Templates Backend', () => {
    let app: express.Express;
    beforeAll(() => {
        app = express();
        const controller = new SiteTemplateController();
        app.use('/', controller.createRouter());
    });

    it('should load site templates properly', async () => {
        const templates = await getSiteTemplates();
        // not super comprehensive
        expect(templates.meta.screenshot_dimensions).toHaveProperty('width');
        expect(templates.meta.screenshot_dimensions).toHaveProperty('height');
        expect(Object.entries(templates.templates).length).toBeTruthy();
    });

    it('should respond to plain site template requests', async () => {
        await request(app).get('/api/siteTemplates').expect(200).expect('Content-Type', /json/);
    });
});
