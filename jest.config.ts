/**
 * For a detailed explanation regarding each configuration property, visit:
 * https://jestjs.io/docs/configuration
 */

import type {Config} from 'jest';

const config: Config = {
    clearMocks: true,
    collectCoverage: true,
    coverageDirectory: 'coverage',
    coverageProvider: 'v8',
    extensionsToTreatAsEsm: ['.ts'],

    // A map from regular expressions to module names or to arrays of module names that allow to stub out resources with a single module
    // https://stackoverflow.com/questions/73735202/typescript-jest-imports-with-js-extension-cause-error-cannot-find-module
    'moduleNameMapper': {
        '^(\\.\\.?\\/.+)\\.js$': '$1',
    },
    roots: [
        'test',
    ],
    testMatch: [
        '**/**.[jt]s',
    ],

    // https://gist.github.com/danpetitt/37f5c966886f54e457ece4b08d66e404
    'transform': {
        '^.+\\.(mt|t|cj|j)s$': [
            'ts-jest',
            {
                useESM: true,
                tsconfig: 'tsconfig.tests.json',
                diagnostics: {
                    ignoreCodes: [1343],
                },
                // https://github.com/ThomZz/ts-jest-mock-import-meta
                astTransformers: {
                    before: [
                        {
                            path: 'node_modules/ts-jest-mock-import-meta',
                        },
                    ],
                },
            },
        ],
    },
    preset: 'ts-jest/presets/default-esm',
};

export default config;
