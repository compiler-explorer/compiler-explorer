import {runFrontendTest} from './utils';


describe('Motd testing', () => {
    before(() => {
        cy.visit('/');
    });

    runFrontendTest('motd');
});
