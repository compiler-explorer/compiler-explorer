// Copyright (c) 2026, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import '../../static/global';

// ──────────────────────────────────────────────────────────────────────────────
// Console output stubbing / assertion
// ──────────────────────────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────────────────────────
// Page visit helpers
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Visit the app and stub console output. Use this as the standard
 * `beforeEach` for any test that needs a fresh page load.
 */
export function visitPage() {
    cy.visit('/', {
        onBeforeLoad: win => {
            stubConsoleOutput(win);
        },
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GoldenLayout pane discovery
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Find a GoldenLayout pane by matching text in its visible tab title.
 * Each pane lives in an `.lm_item.lm_stack` with a `.lm_title` span in its header.
 * Returns the `.lm_content` element within the matching stack.
 */
export function findPane(titleMatch: string) {
    return cy.contains('span.lm_title:visible', titleMatch).closest('.lm_item.lm_stack').find('.lm_content');
}

// ──────────────────────────────────────────────────────────────────────────────
// Source editor helpers
// ──────────────────────────────────────────────────────────────────────────────

/** Get the source editor pane's Monaco editor (matched by "source" in the tab title). */
export function sourceEditor() {
    return findPane('source').find('.monaco-editor');
}

/** Wait for the page to load with the source editor visible. */
export function waitForEditors() {
    sourceEditor().should('be.visible');
}

/**
 * Sets content in a Monaco editor via the editor API.
 *
 * Uses `window.monaco.editor.getEditors()` to locate the live editor
 * instance and calls `model.setValue()` directly, bypassing DOM input
 * handling entirely. This is resilient to changes in Monaco's input
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
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Monaco text assertions
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Asserts that a Monaco editor's view-lines contain the given text.
 * Handles Monaco's non-breaking space (U+00A0) rendering internally.
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

// ──────────────────────────────────────────────────────────────────────────────
// Compiler pane helpers
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Get the first compiler pane's Monaco editor.
 * The compiler tab title contains "Editor #" (e.g. "x86-64 gcc (Editor #1, Compiler #1)").
 */
export function compilerOutput() {
    return findPane('Editor #').find('.monaco-editor');
}

/** Get the first compiler pane's content area (for toolbar elements like filters and options). */
export function compilerPane() {
    return findPane('Editor #');
}

/** Wait for editors and verify the default code has compiled (looks for "square" in output). */
export function setupAndWaitForCompilation() {
    waitForEditors();
    monacoEditorTextShouldContain(compilerOutput(), 'square');
}

/**
 * Get all visible compiler tab title elements (`span.lm_title`) whose title
 * contains "Editor #". Returns elements in DOM order.
 */
export function allCompilerTabs() {
    return cy.get('span.lm_title:visible').filter(':contains("Editor #")');
}

/**
 * Get the Monaco editor within a specific compiler pane, given its tab element.
 */
export function compilerEditorFromTab($tab: JQuery<HTMLElement>) {
    return cy.wrap($tab).closest('.lm_item.lm_stack').find('.lm_content .monaco-editor');
}

/**
 * Get the content area of a specific compiler pane, given its tab element.
 */
export function compilerPaneFromTab($tab: JQuery<HTMLElement>) {
    return cy.wrap($tab).closest('.lm_item.lm_stack').find('.lm_content');
}

/**
 * Get the last visible compiler tab's content area.
 * Useful for targeting the most recently added compiler pane.
 */
export function lastCompilerContent() {
    return cy
        .get('span.lm_title:visible')
        .filter(':contains("Editor #")')
        .last()
        .closest('.lm_item.lm_stack')
        .find('.lm_content');
}

// ──────────────────────────────────────────────────────────────────────────────
// Adding panes
// ──────────────────────────────────────────────────────────────────────────────

/** Add a new compiler from the source editor's "Add new" dropdown. */
export function addCompilerFromEditor() {
    findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
    cy.get('[data-cy="new-add-compiler-btn"]:visible').first().click();
}

/** Clone/add a compiler from the compiler pane's "Add new" dropdown. */
export function addCompilerFromCompilerPane() {
    cy.get('[data-cy="new-compiler-dropdown-btn"]:visible').first().click();
    cy.get('[data-cy="new-add-compiler-btn"]:visible').first().click();
}

/** Open the conformance view from the source editor's "Add new" dropdown. */
export function openConformanceView() {
    findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
    cy.get('[data-cy="new-conformance-btn"]:visible').first().click();
    cy.get('.conformance-wrapper:visible', {timeout: 5000}).should('exist');
}

/** Get the conformance view's content area. */
export function conformancePane() {
    return cy.get('.conformance-wrapper:visible').closest('.lm_content');
}

/**
 * Add a compiler to the conformance view and select it from the TomSelect picker.
 * Clicks "Add compiler", opens the TomSelect dropdown, and picks the first option.
 */
export function addConformanceCompiler() {
    conformancePane().find('button.add-compiler').click();
    conformancePane().find('.ts-control').last().click();
    cy.get('.ts-dropdown .ts-dropdown-content .option:visible').first().click();
}

/** Open the diff view from the top-level "Add..." menu. */
export function openDiffView() {
    cy.get('#addDropdown').click();
    cy.get('#add-diff:visible').click();
    cy.contains('span.lm_title:visible', 'Diff', {timeout: 5000}).should('exist');
}

// ──────────────────────────────────────────────────────────────────────────────
// Legacy / compatibility
// ──────────────────────────────────────────────────────────────────────────────

/**
 * @deprecated This function doesn't actually clear intercepts despite its name.
 * Retained for backward compatibility with claude-explain.cy.ts.
 */
export function clearAllIntercepts() {
    cy.window().then((win: Cypress.AUTWindow) => {
        win.compilerExplorerOptions = {} as any;
    });
}
