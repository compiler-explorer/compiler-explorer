import '../../static/global';

export function stubConsoleOutput(win: Cypress.AUTWindow) {
    cy.stub(win.console, 'log').as('consoleLog');
    cy.stub(win.console, 'warn').as('consoleWarn');
    cy.stub(win.console, 'error').as('consoleError');
}

export function assertNoConsoleOutput() {
    cy.get('@consoleLog').should('not.be.called');
    cy.get('@consoleWarn').should('not.be.called');
    cy.get('@consoleError').should('not.be.called');
}

/**
 * Clear all network intercepts to prevent accumulation
 */
export function clearAllIntercepts() {
    // Clear any existing intercepts by visiting a clean page and resetting
    cy.window().then((win: Cypress.AUTWindow) => {
        // Reset any cached state
        win.compilerExplorerOptions = {};
    });
}

/**
 * Sets content in a Monaco editor via the editor API.
 *
 * Uses `window.monaco.editor.getModels()` to access the underlying text
 * model directly, bypassing DOM input handling entirely.  This is resilient
 * to changes in Monaco's input strategy (textarea vs EditContext).
 *
 * @param content - The code content to set
 * @param editorIndex - Which editor model to target (default: 0 for first)
 */
export function setMonacoEditorContent(content: string, editorIndex = 0) {
    cy.get('.monaco-editor').should('be.visible');

    cy.window().then((win: Cypress.AUTWindow) => {
        const models = win.monaco.editor.getModels();
        expect(models.length, 'at least one Monaco model should exist').to.be.greaterThan(editorIndex);
        models[editorIndex].setValue(content);
    });

    // Wait for compilation to complete after content change (if compiler exists)
    cy.get('body').then(($body: JQuery<HTMLElement>) => {
        if ($body.find('.compiler-wrapper').length > 0) {
            cy.get('.compiler-wrapper').should('not.have.class', 'compiling');
        }
        // If no compiler wrapper exists yet, that's fine - compilation will happen when compiler is added
    });
}
