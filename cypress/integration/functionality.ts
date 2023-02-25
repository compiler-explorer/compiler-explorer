import {runFrontendTest} from '../support/utils.js';

describe('Motd testing', () => {
    before(() => {
        cy.visit('/');
    });

    runFrontendTest('motd');
});
