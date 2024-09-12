// eslint-disable-next-line n/no-unpublished-import
import {defineConfig} from 'cypress';

// eslint-disable-next-line import/no-default-export
export default defineConfig({
    viewportWidth: 1000,
    viewportHeight: 700,
    e2e: {
        baseUrl: 'http://127.0.0.1:10240/',
        supportFile: 'cypress/support/utils.ts',
    },
});
