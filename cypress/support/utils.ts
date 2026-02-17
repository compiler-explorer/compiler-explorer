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
 * Asserts that a Monaco editor's view-lines contain the given text.
 *
 * Retries automatically until the timeout (default: 10000ms).
 * Handles Monaco's non-breaking space rendering internally.
 *
 * @param monacoEditorSelector - A Cypress chainable pointing to a .monaco-editor element
 * @param expectedText - The text substring to look for
 * @param timeout - Optional timeout in ms (default: 10000)
 */
export function monacoEditorTextShouldContain(
    monacoEditorSelector: Cypress.Chainable<JQuery<HTMLElement>>,
    expectedText: string,
    timeout = 10000,
) {
    monacoEditorSelector.find('.view-lines', {timeout}).should($el => {
        const text = $el.text().replaceAll('\u00a0', ' ');
        expect(text).to.include(expectedText);
    });
}

/**
 * Asserts that a Monaco editor's view-lines do NOT contain the given text.
 *
 * @param monacoEditorSelector - A Cypress chainable pointing to a .monaco-editor element
 * @param unexpectedText - The text substring that should be absent
 * @param timeout - Optional timeout in ms (default: 10000)
 */
export function monacoEditorTextShouldNotContain(
    monacoEditorSelector: Cypress.Chainable<JQuery<HTMLElement>>,
    unexpectedText: string,
    timeout = 10000,
) {
    monacoEditorSelector.find('.view-lines', {timeout}).should($el => {
        const text = $el.text().replaceAll('\u00a0', ' ');
        expect(text).to.not.include(unexpectedText);
    });
}

/**
 * Clear all network intercepts to prevent accumulation
 */
export function clearAllIntercepts() {
    // Clear any existing intercepts by visiting a clean page and resetting
    cy.window().then((win: Cypress.AUTWindow) => {
        // Reset any cached state
        win.compilerExplorerOptions = {} as any;
    });
}

/**
 * Sets content in a Monaco editor via the editor API.
 *
 * Uses `window.monaco.editor.getEditors()` to locate the live editor
 * instance and calls `model.setValue()` directly, bypassing DOM input
 * handling entirely.  This is resilient to changes in Monaco's input
 * strategy (textarea vs EditContext).
 *
 * @param content - The code content to set
 * @param editorIndex - Which editor to target (default: 0 for first)
 */
export function setMonacoEditorContent(content: string, editorIndex = 0) {
    cy.get('.monaco-editor').should('be.visible');

    cy.window().then((win: Cypress.AUTWindow) => {
        const editors = win.monaco.editor.getEditors();
        expect(editors.length, 'at least one Monaco editor should exist').to.be.greaterThan(editorIndex);
        editors[editorIndex].getModel()?.setValue(content);
    });

    // Wait for compilation to complete after content change (if compiler exists)
    cy.get('body').then(($body: JQuery<HTMLElement>) => {
        if ($body.find('.compiler-wrapper').length > 0) {
            cy.get('.compiler-wrapper').should('not.have.class', 'compiling');
        }
        // If no compiler wrapper exists yet, that's fine - compilation will happen when compiler is added
    });
}
