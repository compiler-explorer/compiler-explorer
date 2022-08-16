function runFrontendTest(name) {
    it(name, () => {
        cy.window().then(win => {
            return win.compilerExplorerFrontendTesting.run(name);
        });
    });
}

describe('Frontendtestresults', () => {
    before(() => {
        cy.visit('/');
    });

    runFrontendTest('HelloWorld');
});

describe('Motd testing', () => {
    before(() => {
        cy.visit('/');
    });

    runFrontendTest('motd');
});
