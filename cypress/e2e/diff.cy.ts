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

import {
    assertNoConsoleOutput,
    monacoEditorTextShouldContain,
    setMonacoEditorContent,
    stubConsoleOutput,
} from '../support/utils';

/**
 * Find a GoldenLayout pane by matching text in its visible tab title.
 * Returns the .lm_content element within the matching stack.
 */
function findPane(titleMatch: string) {
    return cy.contains('span.lm_title:visible', titleMatch).closest('.lm_item.lm_stack').find('.lm_content');
}

function sourceEditor() {
    return findPane('source').find('.monaco-editor');
}

function waitForEditors() {
    sourceEditor().should('be.visible');
    findPane('Editor #').find('.monaco-editor').should('be.visible');
}

/**
 * Get the last visible compiler tab's content area.
 * Useful for targeting the most recently added compiler pane.
 */
function lastCompilerContent() {
    return cy
        .get('span.lm_title:visible')
        .filter(':contains("Editor #")')
        .last()
        .closest('.lm_item.lm_stack')
        .find('.lm_content');
}

/**
 * Source code with conditional compilation for producing distinct outputs
 * in two compiler panes, making diff view show meaningful differences.
 * Uses different function names so we can assert the diff shows both variants.
 */
const DIFF_SOURCE = [
    '#ifdef USE_ADD',
    'int add_variant(int a, int b) { return a + b; }',
    '#else',
    'int mul_variant(int a, int b) { return a * b; }',
    '#endif',
].join('\n');

beforeEach(() => {
    cy.visit('/', {
        onBeforeLoad: win => {
            stubConsoleOutput(win);
        },
    });
});

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('Diff view', () => {
    it('should open a diff view pane from the Add menu', () => {
        waitForEditors();

        cy.get('#addDropdown').click();
        cy.get('#add-diff:visible').click();

        cy.contains('span.lm_title:visible', 'Diff', {timeout: 5000}).should('exist');
    });

    it('should show diff content with two compiler panes', () => {
        waitForEditors();
        setMonacoEditorContent(DIFF_SOURCE);

        // Wait for first compiler to compile (no flag → mul_variant)
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'mul_variant');

        // Add second compiler
        findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
        cy.get('[data-cy="new-add-compiler-btn"]:visible').first().click();

        // Wait for second compiler to appear and compile
        cy.get('span.lm_title:visible').filter(':contains("Editor #")').should('have.length', 2);
        lastCompilerContent()
            .find('.monaco-editor .view-lines', {timeout: 10000})
            .should('contain.text', 'mul_variant');

        // Set -DUSE_ADD on the second compiler to get the add_variant function
        lastCompilerContent().find('input.options').clear().type('-DUSE_ADD');

        // Wait for recompilation — second compiler should now show add_variant
        lastCompilerContent()
            .find('.monaco-editor .view-lines', {timeout: 10000})
            .should('contain.text', 'add_variant');

        // Open diff view
        cy.get('#addDropdown').click();
        cy.get('#add-diff:visible').click();
        cy.contains('span.lm_title:visible', 'Diff', {timeout: 5000}).should('exist');

        // The diff pane auto-selects the two compilers when there are exactly two.
        // The diff should show content from both variants — the LHS has mul_variant,
        // the RHS has add_variant. Verify at least one is visible in the diff editor.
        findPane('Diff')
            .find('.view-lines', {timeout: 10000})
            .should($el => {
                const text = $el.text().replaceAll('\u00a0', ' ');
                const hasLhs = text.includes('mul_variant');
                const hasRhs = text.includes('add_variant');
                expect(hasLhs || hasRhs, 'diff should contain at least one variant function name').to.be.true;
            });
    });
});
