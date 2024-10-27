import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {fixupConfigRules, fixupPluginRules} from '@eslint/compat';
import {FlatCompat} from '@eslint/eslintrc';
import js from '@eslint/js';
import typescriptEslint from '@typescript-eslint/eslint-plugin';
import _import from 'eslint-plugin-import';
import jsdoc from 'eslint-plugin-jsdoc';
import n from 'eslint-plugin-n';
import prettier from 'eslint-plugin-prettier';
import promise from 'eslint-plugin-promise';
import unicorn from 'eslint-plugin-unicorn';
import unusedImports from 'eslint-plugin-unused-imports';
import globals from 'globals';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
    baseDirectory: __dirname,
    recommendedConfig: js.configs.recommended,
    allConfig: js.configs.all,
});

// eslint-disable-next-line import/no-default-export
export default [
    {
        ignores: [
            '**/coverage',
            '**/docs',
            '**/etc',
            '**/examples',
            '**/out',
            '**/views',
            'lib/asm-docs/generated/asm-docs-*',
            'etc/scripts/docenizer/vendor/jvms.html',
        ],
    },
    ...fixupConfigRules(
        compat.extends(
            'eslint:recommended',
            'plugin:import/recommended',
            'plugin:n/recommended',
            'plugin:unicorn/recommended',
            'prettier',
            'plugin:@typescript-eslint/eslint-recommended',
            'plugin:@typescript-eslint/recommended',
            'plugin:import/typescript',
        ),
    ),
    {
        plugins: {
            import: fixupPluginRules(_import),
            jsdoc,
            n: fixupPluginRules(n),
            promise,
            prettier,
            unicorn: fixupPluginRules(unicorn),
            '@typescript-eslint': fixupPluginRules(typescriptEslint),
            'unused-imports': unusedImports,
        },
        languageOptions: {
            globals: {
                ...globals.node,
                ...globals.browser,
                BigInt: true,
            },
            ecmaVersion: 2020,
            sourceType: 'module',
        },
        settings: {
            node: {
                tryExtensions: ['.js', '.ts'],
            },
            'import/parsers': {
                '@typescript-eslint/parser': ['.ts', '.tsx'],
            },
            'import/resolver': 'typescript',
        },
        rules: {
            'prettier/prettier': 'error',
            'comma-dangle': [
                'error',
                {
                    arrays: 'always-multiline',
                    objects: 'always-multiline',
                    imports: 'always-multiline',
                    exports: 'always-multiline',
                    functions: 'always-multiline',
                },
            ],
            'eol-last': ['error', 'always'],
            eqeqeq: ['error', 'smart'],
            'import/default': 'off',
            'import/first': 'error',
            'import/newline-after-import': 'error',
            'import/no-absolute-path': 'error',
            'import/no-default-export': 'error',
            'import/no-deprecated': 'error',
            'import/no-mutable-exports': 'error',
            'import/no-named-as-default-member': 'off',
            'import/no-self-import': 'error',
            'import/no-useless-path-segments': 'error',
            'import/no-webpack-loader-syntax': 'error',
            'import/no-unresolved': 'off',
            'import/order': [
                'error',
                {
                    alphabetize: {
                        order: 'asc',
                        caseInsensitive: true,
                    },
                    'newlines-between': 'always',
                },
            ],
            'max-len': [
                'error',
                120,
                {
                    ignoreRegExpLiterals: true,
                },
            ],
            'max-statements': ['error', 100],
            'no-console': 'error',
            'no-control-regex': 0,
            'no-duplicate-imports': 'error',
            'no-useless-call': 'error',
            'no-useless-computed-key': 'error',
            'no-useless-concat': 'error',
            '@typescript-eslint/no-useless-constructor': 'error',
            'no-useless-escape': 'error',
            'no-useless-rename': 'error',
            'no-useless-return': 'error',
            'no-empty': [
                'error',
                {
                    allowEmptyCatch: true,
                },
            ],
            'quote-props': ['error', 'as-needed'],
            quotes: [
                'error',
                'single',
                {
                    allowTemplateLiterals: true,
                    avoidEscape: true,
                },
            ],
            semi: ['error', 'always'],
            'space-before-function-paren': [
                'error',
                {
                    anonymous: 'always',
                    asyncArrow: 'always',
                    named: 'never',
                },
            ],
            'keyword-spacing': [
                'error',
                {
                    after: true,
                },
            ],
            yoda: [
                'error',
                'never',
                {
                    onlyEquality: true,
                },
            ],
            'prefer-const': [
                'error',
                {
                    destructuring: 'all',
                },
            ],
            'jsdoc/check-alignment': 'error',
            'jsdoc/check-param-names': 'error',
            'jsdoc/check-syntax': 'error',
            'jsdoc/check-tag-names': 'off',
            'jsdoc/check-types': 'error',
            'jsdoc/empty-tags': 'error',
            'jsdoc/require-hyphen-before-param-description': 'error',
            'jsdoc/valid-types': 'error',
            'no-multiple-empty-lines': [
                'error',
                {
                    max: 1,
                    maxBOF: 0,
                    maxEOF: 0,
                },
            ],
            'n/no-process-exit': 'off',
            'n/no-missing-import': [
                'error',
                {
                    allowModules: ['monaco-editor'],
                },
            ],
            'promise/catch-or-return': 'error',
            'promise/no-new-statics': 'error',
            'promise/no-return-wrap': 'error',
            'promise/param-names': 'error',
            'promise/valid-params': 'error',
            'sort-imports': [
                'error',
                {
                    ignoreCase: true,
                    ignoreDeclarationSort: true,
                },
            ],
            'unicorn/catch-error-name': 'off',
            'unicorn/consistent-function-scoping': 'off',
            'unicorn/empty-brace-spaces': 'off',
            'unicorn/no-fn-reference-in-iterator': 'off',
            'unicorn/no-hex-escape': 'off',
            'unicorn/no-null': 'off',
            'unicorn/no-reduce': 'off',
            'unicorn/numeric-separators-style': 'off',
            'unicorn/prefer-add-event-listener': 'off',
            'unicorn/prefer-flat-map': 'error',
            'unicorn/prefer-optional-catch-binding': 'off',
            'unicorn/prefer-node-protocol': 'off',
            'unicorn/prefer-number-properties': 'off',
            'unicorn/prefer-string-slice': 'off',
            'unicorn/prevent-abbreviations': 'off',
            'unicorn/prefer-ternary': 'off',
            'unicorn/prefer-array-some': 'off',
            'unicorn/prefer-spread': 'off',
            'unicorn/no-lonely-if': 'off',
            'unicorn/no-array-reduce': 'off',
            'unicorn/prefer-array-flat': 'off',
            'unicorn/no-array-callback-reference': 'off',
            'unicorn/prefer-switch': 'off',
            'unicorn/no-static-only-class': 'off',
            'unicorn/no-process-exit': 'off',
            'unicorn/no-useless-undefined': [
                'error',
                {
                    checkArguments: false,
                },
            ],
            'unused-imports/no-unused-imports': 'error',
            '@typescript-eslint/no-empty-function': 'off',
            '@typescript-eslint/no-unused-vars': 'off',
            '@typescript-eslint/no-explicit-any': 'off',
            'unicorn/prefer-at': 'off',
            'unicorn/prefer-negative-index': 'off',
        },
    },
];
