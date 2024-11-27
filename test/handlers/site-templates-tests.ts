import express from 'express';
import request from 'supertest';
import {beforeAll, describe, expect, it} from 'vitest';

import {SiteTemplateController} from '../../lib/handlers/api/site-template-controller.js';
import {getSiteTemplates} from '../../lib/handlers/site-templates.js';

describe('Site Templates Backend', () => {
    let app: express.Express;
    beforeAll(() => {
        app = express();
        const controller = new SiteTemplateController();
        app.use('/api/siteTemplates', controller.getSiteTemplates.bind(controller));
    });

    it('should load site templates properly', async () => {
        const templates = await getSiteTemplates();
        // not super comprehensive
        expect(templates.meta).toHaveProperty('meta.screenshot_dimensions');
        expect(Object.entries(templates.templates).length).toBeTruthy();
    });

    it('should respond to plain site template requests', async () => {
        await request(app).get('/api/siteTemplates').expect(200).expect('Content-Type', /json/);
    });
});
