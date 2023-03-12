import {defineConfig} from 'cypress';

export default defineConfig({
    e2e: {
        baseUrl: 'http://127.0.0.1:10240/',
        supportFile: 'cypress/support/utils.ts',
    },
});
