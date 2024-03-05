import {defineConfig} from 'vitest/config';

export default defineConfig({
    test: {
        include: [
            "test/**/*.ts",
        ],
        setupFiles: ['/test/_setup-fake-aws.ts', '/test/_setup-log.ts'],
    },
});
