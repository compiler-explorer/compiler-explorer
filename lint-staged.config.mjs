// eslint-disable-next-line import/no-default-export
export default {
    '*.ts': [() => 'npm run ts-check', 'npm run lint'],
    '*.js': ['npm run format-files --', 'npm run lint'],
    '*.{html,md}': ['npm run lint'],
};
