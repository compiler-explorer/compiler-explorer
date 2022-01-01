describe('Frontendtestresults', () => {
    it('Startup', () => {
        cy.visit('/');
        cy.get('title').should('have.html', 'Compiler Explorer');
        cy.window().then((win) => {
            if (win.frontendTesting) {
                return win.frontendTesting.run();
            }
        });
    });
});