// This file is in js due to #3514

module.exports = {
    root: true,
    plugins: [
        'promise',
        'requirejs',
    ],
    extends: ['./.eslint-ce-static.yml'],
    rules: {
        'promise/catch-or-return': 'off',
        'promise/no-new-statics': 'error',
        'promise/no-return-wrap': 'error',
        'promise/param-names': 'error',
        'promise/valid-params': 'error',
    },
    overrides: [{
        files: ['*.ts'],
        plugins: [
            'import',
            '@typescript-eslint',
        ],
        extends: [
            './.eslint-ce-static.yml',
            'plugin:@typescript-eslint/eslint-recommended',
            'plugin:@typescript-eslint/recommended',
            'plugin:import/typescript',
        ],
        env: {
            browser: true,
            es6: true,
            node: false,
        },
        parser: '@typescript-eslint/parser',
        parserOptions: {
            sourceType: 'module',
            ecmaVersion: 'latest',
            tsconfigRootDir: __dirname,
            project: './tsconfig.json',
        },
        rules: {
            '@typescript-eslint/no-empty-function': 'off',
            '@typescript-eslint/no-unused-vars': 'off',
            '@typescript-eslint/no-var-requires': 'off', // Needed for now, can't move some
            '@typescript-eslint/no-explicit-any': 'off', // Too much js code still exists
            '@typescript-eslint/ban-ts-comment': 'off', // We need some @ts-ignore at some points
            '@typescript-eslint/no-unnecessary-condition': 'error',
        },
    }],
};
