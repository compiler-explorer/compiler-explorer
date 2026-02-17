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

import {assertNoConsoleOutput, setMonacoEditorContent, stubConsoleOutput} from '../support/utils';

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
}

/**
 * Open the conformance view from the editor's "Add new" dropdown.
 */
function openConformanceView() {
    findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
    cy.get('[data-cy="new-conformance-btn"]:visible').first().click();
    // Wait for the conformance pane's wrapper to appear
    cy.get('.conformance-wrapper:visible', {timeout: 5000}).should('exist');
}

/**
 * Get the conformance view's content area.
 */
function conformancePane() {
    return cy.get('.conformance-wrapper:visible').closest('.lm_content');
}

/**
 * Add a compiler to the conformance view and select it from the TomSelect picker.
 * After clicking "Add compiler", the picker row appears but no compiler is selected.
 * We click the TomSelect input to open the dropdown, then select the first option.
 */
function addAndSelectCompiler() {
    conformancePane().find('button.add-compiler').click();

    // Click the TomSelect input to open the compiler dropdown.
    // TomSelect wraps the <select> with a .ts-wrapper; clicking the .ts-control opens it.
    conformancePane().find('.ts-control').last().click();

    // Select the first compiler option in the TomSelect dropdown
    cy.get('.ts-dropdown .ts-dropdown-content .option:visible').first().click();
}

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

describe('Conformance view', () => {
    it('should open a conformance view pane', () => {
        waitForEditors();
        openConformanceView();

        cy.get('.conformance-wrapper:visible').should('exist');
    });

    it('should show a pass indicator for code that compiles cleanly', () => {
        waitForEditors();
        setMonacoEditorContent('int square(int n) { return n * n; }');

        openConformanceView();
        addAndSelectCompiler();

        // A successful compilation shows a green check-circle icon.
        // The icon class is set by CompilerService.handleCompilationStatus:
        //   code 1 or 2 → fa-check-circle, code 3 → fa-times-circle
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');
    });

    it('should show a fail indicator for code with static_assert failure', () => {
        waitForEditors();
        setMonacoEditorContent('static_assert(false, "deliberate failure");');

        openConformanceView();
        addAndSelectCompiler();

        // A failed compilation shows a red times-circle icon
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });

    it('should update status when source code changes', () => {
        waitForEditors();
        setMonacoEditorContent('int valid_func(int x) { return x; }');

        openConformanceView();
        addAndSelectCompiler();

        // Should initially pass
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');

        // Change to invalid code
        setMonacoEditorContent('static_assert(false, "now it fails");');

        // Should now show failure
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });

    it('should support multiple compilers with different options', () => {
        waitForEditors();
        // Code that passes normally but fails with -DFAIL_ME
        setMonacoEditorContent(
            [
                '#ifdef FAIL_ME',
                'static_assert(false, "conditional failure");',
                '#else',
                'int all_good(int x) { return x; }',
                '#endif',
            ].join('\n'),
        );

        openConformanceView();

        // Add first compiler — should pass (no -D flag)
        addAndSelectCompiler();
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');

        // Add second compiler and select it
        addAndSelectCompiler();

        // Set -DFAIL_ME on the second compiler's options input
        conformancePane().find('.conformance-options').last().clear().type('-DFAIL_ME');

        // Should now have both a pass and a fail indicator
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });
});
