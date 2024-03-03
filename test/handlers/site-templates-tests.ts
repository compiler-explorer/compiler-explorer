import {getSiteTemplates, loadSiteTemplates} from '../../lib/handlers/site-templates.js';

describe('Site Templates Backend', () => {
    beforeAll(() => {
        loadSiteTemplates('etc/config');
    });

    it('should load site templates properly', () => {
        const templates = getSiteTemplates();
        // not super comprehensive
        expect(templates.meta['meta.screenshot_dimentions'] !== undefined).toBeTruthy();
        expect(Object.entries(templates.templates).length > 0).toBeTruthy();
    });
});
