module.exports = {
    trailingComma: 'all',
    printWidth: 120,
    singleQuote: true,
    arrowParens: 'avoid',
    tabWidth: 4,
    bracketSpacing: false,
    overrides: [
        {
            files: '*.{yml,json}',
            options: {
                tabWidth: 2,
            },
        },
    ],
};
