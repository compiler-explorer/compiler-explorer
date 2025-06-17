export default {
    '*.ts': ['npm run lint', () => 'npm run ts-check'],
    '*.{html,md,js}': ['npm run lint'],
};
