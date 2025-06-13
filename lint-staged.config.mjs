export default {
    '*.ts': ['npm run lint', () => 'npm run ts-check', () => 'npm run check-imports'],
    '*.{html,md,js}': ['npm run lint'],
};
