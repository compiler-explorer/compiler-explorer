import { suppressConsoleLog } from '../lib/logger';

// this hook will run once before any tests are executed
before(() => {
    suppressConsoleLog();
});
