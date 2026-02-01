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
 * Sets content in Monaco editor using keyboard input.
 *
 * Previous versions used a synthetic ClipboardEvent('paste') dispatched to
 * the textarea, but Monaco 0.53+ may use the native EditContext API instead
 * of a textarea for input handling, which ignores synthetic paste events.
 *
 * Using Cypress's cy.type() simulates real keyboard input at the browser
 * level, which works regardless of Monaco's internal input method.
 *
 * @param content - The code content to set
 * @param editorIndex - Which editor to target (default: 0 for first editor)
 */
export function setMonacoEditorContent(content: string, editorIndex = 0) {
    // Wait for Monaco editor to be visible in DOM
    cy.get('.monaco-editor').should('be.visible');

    // Select all existing content and delete it, then type the new content.
    // Escape special Cypress characters in the content (braces are used for
    // special keys like {enter}), so we need to wrap literal braces.
    const escaped = content.replace(/\{/g, '{{}').replace(/\n/g, '{enter}');

    cy.get('.monaco-editor textarea')
        .eq(editorIndex)
        .focus()
        .type('{ctrl}a{del}', {force: true})
        .type(escaped, {force: true, delay: 0});

    // Wait for compilation to complete after content change (if compiler exists)
    cy.get('body').then(($body: JQuery<HTMLElement>) => {
        if ($body.find('.compiler-wrapper').length > 0) {
            cy.get('.compiler-wrapper').should('not.have.class', 'compiling');
        }
        // If no compiler wrapper exists yet, that's fine - compilation will happen when compiler is added
    });
}
