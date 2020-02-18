const logger = require('../lib/logger');

// this hook will run once before any tests are executed
before(() => {
    logger.suppressConsoleLog();
});
