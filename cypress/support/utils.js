export function runFrontendTest(name) {
    it(name, () => {
        cy.window().then(win => {
            return win.compilerExplorerFrontendTesting.run(name);
        });
    });
}
