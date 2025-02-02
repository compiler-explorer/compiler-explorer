import {defineConfig} from 'cypress';

export default defineConfig({
    viewportWidth: 1000,
    viewportHeight: 700,
    e2e: {
        baseUrl: 'http://127.0.0.1:10240/',
        supportFile: 'cypress/support/utils.ts',
    },
});
