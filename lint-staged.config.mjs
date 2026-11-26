// Note: the license-header check is NOT wired in here. Like check-frontend-imports,
// it runs as a single fast full-tree scan in the husky pre-commit hook (and in CI /
// `npm run check` / `make pre-commit`), which is cheap and catches everything rather
// than only staged files.
export default {
    '*.ts': ['npm run lint', () => 'npm run ts-check', 'cross-env SKIP_EXPENSIVE_TESTS=true npx vitest related --run'],
    '*.{html,md,js}': ['npm run lint'],
    '*.properties': [() => 'npm run test:props'],
};
