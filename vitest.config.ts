// eslint-disable-next-line n/no-unpublished-import
import {defineConfig} from 'vitest/config';

// eslint-disable-next-line import/no-default-export
export default defineConfig({
    test: {
        coverage: {
            provider: 'v8',
            reporter: ['text', 'json', 'html'],
        },
        include: ['test/**/*.ts'],
        exclude: ['test/_*.ts', 'test/utils.ts'],
        setupFiles: ['/test/_setup-fake-aws.ts', '/test/_setup-log.ts'],
    },
});
