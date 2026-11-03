import path from 'node:path';

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
                    // Monaco's package "exports" map no longer allows deep esm/ paths; see webpack.config.esm.ts.
                    alias: {
                        'monaco-editor/esm': path.resolve(__dirname, 'node_modules/monaco-editor/esm'),
                    },
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
