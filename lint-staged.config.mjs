export default {
    '*.ts': [
        'npm run lint',
        'node ./etc/scripts/check-license-headers.js',
        () => 'npm run ts-check',
        'cross-env SKIP_EXPENSIVE_TESTS=true npx vitest related --run',
    ],
    '*.{html,md,js}': ['npm run lint'],
    '*.{js,mjs,cjs}': ['node ./etc/scripts/check-license-headers.js'],
    '*.properties': [() => 'npm run test:props'],
};
