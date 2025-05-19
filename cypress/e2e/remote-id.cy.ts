import {runFrontendTest} from '../support/utils';

describe('RemoteId testing', () => {
    before(() => {
        cy.visit('/');
    });

    runFrontendTest('remoteId');
});
