// Copyright (c) 2022, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// This file is in js due to #3514
module.exports = {
    root: true,
    plugins: ['promise', 'requirejs', 'unused-imports'],
    extends: ['./.eslint-ce-static.yml'],
    rules: {
        'promise/catch-or-return': 'off',
        'promise/no-new-statics': 'error',
        'promise/no-return-wrap': 'error',
        'promise/param-names': 'error',
        'promise/valid-params': 'error',
    },
    overrides: [
        {
            files: ['*.ts'],
            plugins: ['import', '@typescript-eslint'],
            extends: [
                './.eslint-ce-static.yml',
                'plugin:@typescript-eslint/eslint-recommended',
                'plugin:@typescript-eslint/recommended',
                'plugin:import/recommended',
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
                'import/no-unresolved': 'off',
                'node/no-missing-imports': 'off',
                'unused-imports/no-unused-imports': 'error',
                '@typescript-eslint/await-thenable': 'error',
                '@typescript-eslint/no-empty-function': 'off',
                '@typescript-eslint/no-unused-vars': 'off',
                '@typescript-eslint/no-var-requires': 'off', // Needed for now, can't move some
                '@typescript-eslint/no-explicit-any': 'off', // Too much js code still exists
                '@typescript-eslint/ban-ts-comment': ['error', {'ts-expect-error': true}],
                '@typescript-eslint/no-unnecessary-condition': 'error',
                '@typescript-eslint/no-unnecessary-type-assertion': 'error',
                '@typescript-eslint/prefer-includes': 'error',
            },
        },
    ],
};
