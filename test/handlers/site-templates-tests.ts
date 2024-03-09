import {beforeAll, describe, expect, it} from 'vitest';

import {getSiteTemplates, loadSiteTemplates} from '../../lib/handlers/site-templates.js';

describe('Site Templates Backend', () => {
    beforeAll(() => {
        loadSiteTemplates('etc/config');
    });

    it('should load site templates properly', () => {
        const templates = getSiteTemplates();
        // not super comprehensive
        expect(templates.meta).toHaveProperty('meta.screenshot_dimentions');
        expect(Object.entries(templates.templates).length).toBeTruthy();
    });
});
