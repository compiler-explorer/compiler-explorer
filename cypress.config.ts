// eslint-disable-next-line node/no-unpublished-import
import {defineConfig} from 'cypress';

// eslint-disable-next-line import/no-default-export
export default defineConfig({
    e2e: {
        baseUrl: 'http://127.0.0.1:10240/',
        supportFile: 'cypress/support/utils.ts',
    },
});
