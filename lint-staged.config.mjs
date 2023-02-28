// eslint-disable-next-line import/no-default-export
export default {
    '*.ts': [() => 'npm run ts-check'],
    '.*(?<!cypress).*.[jt]s': ['npm run format-files --', 'npm run lint-files --'],
    '*.{html,md}': ['npm run format-files --'],
};
