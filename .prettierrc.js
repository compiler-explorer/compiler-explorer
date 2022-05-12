module.exports = {
    trailingComma: 'all',
    printWidth: 120,
    singleQuote: true,
    arrowParens: 'avoid',
    tabWidth: 4,
    bracketSpacing: false,
    proseWrap: 'always',
    overrides: [
        {
            files: '*.{yml,json,md}',
            options: {
                tabWidth: 2,
            },
        },
    ],
};
