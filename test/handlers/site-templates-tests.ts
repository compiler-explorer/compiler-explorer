import {assert} from 'chai';

import {getSiteTemplates, loadSiteTemplates} from '../../lib/handlers/site-templates.js';

describe('Site Templates Backend', () => {
    before(() => {
        loadSiteTemplates('etc/config');
    });

    it('should load site templates properly', () => {
        const templates = getSiteTemplates();
        // not super comprehensive
        assert(templates.meta['meta.screenshot_dimentions'] !== undefined);
        assert(Object.entries(templates.templates).length > 0);
    });
});
