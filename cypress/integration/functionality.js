import {runFrontendTest} from '../support/utils';

describe('Motd testing', () => {
    before(() => {
        cy.visit('/');
    });

    runFrontendTest('motd');
});
