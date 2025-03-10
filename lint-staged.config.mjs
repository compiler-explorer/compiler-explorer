export default {
    '*.ts': ['npm run lint', () => 'npm run ts-check'],
    '*.js': ['npm run format-files --', 'npm run lint'],
    '*.{html,md}': ['npm run lint'],
};
