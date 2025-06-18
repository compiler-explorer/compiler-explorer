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
                    exclude: ['test/_*.ts', 'test/utils.ts', 'test/static'],
                    setupFiles: ['/test/_setup-fake-aws.ts', '/test/_setup-log.ts'],
                },
            },
            {
                test: {
                    name: 'frontend unit',
                    include: ['test/static/**/*.ts'],
                    setupFiles: ['/test/_setup-fake-aws.ts', '/test/_setup-log.ts', '/test/static/_setup-dom.ts'],
                    environment: 'jsdom',
                },
            },
        ],
    },
});
