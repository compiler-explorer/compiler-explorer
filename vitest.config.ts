// eslint-disable-next-line node/no-unpublished-import
import {defineConfig} from 'vitest/config';

// eslint-disable-next-line import/no-default-export
export default defineConfig({
    test: {
        include: ['test/**/*.ts'],
        exclude: ['test/_*.ts', 'test/utils.ts'],
        setupFiles: ['/test/_setup-fake-aws.ts', '/test/_setup-log.ts'],
    },
});
