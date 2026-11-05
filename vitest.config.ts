import {defineConfig} from 'vitest/config';

export default defineConfig({
    test: {
        coverage: {
            provider: 'v8',
            reporter: ['text', 'json', 'html'],
        },
        projects: [
            {
                test: {
                    name: 'unit',
                    include: ['test/**/*.ts'],
                    exclude: ['test/_*.ts', 'test/utils.ts'],
                    setupFiles: ['test/_setup-fake-aws.ts', 'test/_setup-log.ts'],
                },
            },
            {
                resolve: {
                    // Maps the pre-0.56 deep paths (still used by monaco-vim and by editor.ts) onto Monaco's "exports" map.
                    alias: [{find: /^monaco-editor\/esm\/vs\//, replacement: 'monaco-editor/'}],
                },
                test: {
                    name: 'frontend unit',
                    include: ['static/tests/**/*.ts'],
                    exclude: ['static/tests/_*.ts'],
                    setupFiles: ['static/tests/_setup-dom.ts'],
                    environment: 'happy-dom',
                },
            },
        ],
    },
});
