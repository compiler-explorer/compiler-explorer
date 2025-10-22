export default {
    '*.ts': ['npm run lint', () => 'npm run ts-check', 'cross-env SKIP_EXPENSIVE_TESTS=true npx vitest related --run'],
    '*.{html,md,js}': ['npm run lint'],
};
