import {describe, expect, it} from 'vitest';

import {getSiteTemplates} from '../../lib/handlers/site-templates.js';

describe('Site Templates Backend', () => {
    it('should load site templates properly', async () => {
        const templates = await getSiteTemplates();
        // not super comprehensive
        expect(templates.meta).toHaveProperty('meta.screenshot_dimensions');
        expect(Object.entries(templates.templates).length).toBeTruthy();
    });
});
