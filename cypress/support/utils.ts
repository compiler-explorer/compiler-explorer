import '../../static/global';

// Disable the native EditContext API before Monaco initialises.  Monaco 0.53+
// uses EditContext when available (Chrome 133+), replacing the hidden textarea
// with a <div> that receives input through the EditContext interface.  Synthetic
// paste events dispatched via ClipboardEvent no longer reach the editor through
// that path, which breaks Cypress helpers that rely on pasting content into the
// textarea.  Removing the constructor forces Monaco to fall back to its textarea
// input handler.
Cypress.on('window:before:load', (win: Cypress.AUTWindow) => {
    Object.defineProperty(win, 'EditContext', {value: undefined, writable: true, configurable: true});
});

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
 * Sets content in Monaco editor using a synthetic paste event
 * @param content - The code content to set
 * @param editorIndex - Which editor to target (default: 0 for first editor)
 */
export function setMonacoEditorContent(content: string, editorIndex = 0) {
    // Wait for Monaco editor to be visible in DOM
    cy.get('.monaco-editor').should('be.visible');

    // Select all and delete existing content
    cy.get('.monaco-editor textarea').eq(editorIndex).focus().type('{ctrl}a{del}', {force: true});

    // Trigger a paste event with our content
    cy.get('.monaco-editor textarea')
        .eq(editorIndex)
        .then(($element: JQuery<HTMLTextAreaElement>) => {
            const el = $element[0];

            // Create and dispatch a paste event with our data
            const pasteEvent = new ClipboardEvent('paste', {
                bubbles: true,
                cancelable: true,
                clipboardData: new DataTransfer(),
            });

            // Add our text to the clipboard data
            pasteEvent.clipboardData?.setData('text/plain', content);

            // Dispatch the event
            el.dispatchEvent(pasteEvent);
        });

    // Wait for compilation to complete after content change (if compiler exists)
    cy.get('body').then(($body: JQuery<HTMLElement>) => {
        if ($body.find('.compiler-wrapper').length > 0) {
            cy.get('.compiler-wrapper').should('not.have.class', 'compiling');
        }
        // If no compiler wrapper exists yet, that's fine - compilation will happen when compiler is added
    });
}
