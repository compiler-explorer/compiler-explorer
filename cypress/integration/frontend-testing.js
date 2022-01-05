function runFrontendTest(name) {
    it(name, () => {
        cy.window().then((win) => {
            return win.frontendTesting.run(name);
        });
    });
}

describe('Frontendtestresults', () => {
    before(() => {
        cy.visit('/');
    });

    runFrontendTest('HelloWorld');
});
