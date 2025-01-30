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
