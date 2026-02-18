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

/** Get the source editor pane's Monaco editor. */
function sourceEditor() {
    return findPane('source').find('.monaco-editor');
}

/** Wait for the page to load with both source and compiler editors visible. */
function waitForEditors() {
    sourceEditor().should('be.visible');
    findPane('Editor #').find('.monaco-editor').should('be.visible');
}

/**
 * Get all visible compiler tab title elements (`span.lm_title`) corresponding
 * to compiler editors (those whose title contains "Editor #").
 *
 * Note: The returned elements reflect the DOM order; callers must apply their
 * own sorting if a specific ordering by title is required.
 */
function allCompilerTabs() {
    return cy.get('span.lm_title:visible').filter(':contains("Editor #")');
}

/**
 * Get the Monaco editor within a specific compiler pane by its tab element.
 */
function compilerEditorFromTab($tab: JQuery<HTMLElement>) {
    return cy.wrap($tab).closest('.lm_item.lm_stack').find('.lm_content .monaco-editor');
}

/**
 * Get the content area of a specific compiler pane by its tab element.
 */
function compilerPaneFromTab($tab: JQuery<HTMLElement>) {
    return cy.wrap($tab).closest('.lm_item.lm_stack').find('.lm_content');
}

/**
 * Source code using #ifdef to produce different function names based on -D flags.
 * This lets us verify that two compiler panes with different options produce
 * different assembly output using only the single available compiler (gdefault).
 */
const CONDITIONAL_SOURCE = [
    '#ifdef VARIANT_A',
    'int variant_a_func(int x) { return x + 1; }',
    '#elif defined(VARIANT_B)',
    'int variant_b_func(int x) { return x - 1; }',
    '#else',
    'int default_func(int x) { return x * 2; }',
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

describe('Multi-compiler panes', () => {
    beforeEach(() => {
        waitForEditors();
        setMonacoEditorContent(CONDITIONAL_SOURCE);
    });

    it('should add a second compiler pane from the editor dropdown', () => {
        findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
        cy.get('[data-cy="new-add-compiler-btn"]:visible').first().click();

        allCompilerTabs().should('have.length', 2);
    });

    it('should produce different output with different -D flags', () => {
        // First compiler: default (no flags) → should show default_func
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'default_func');

        // Add second compiler
        findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
        cy.get('[data-cy="new-add-compiler-btn"]:visible').first().click();
        allCompilerTabs().should('have.length', 2);

        // Set -DVARIANT_A on the second compiler pane.
        // The second compiler tab will be the last one, so we target its options input.
        allCompilerTabs()
            .last()
            .then($tab => {
                compilerPaneFromTab($tab).find('input.options').clear().type('-DVARIANT_A');

                // Second compiler should show variant_a_func
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'variant_a_func');
                });
            });

        // First compiler should still show default_func (unchanged)
        allCompilerTabs()
            .first()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'default_func');
                });
            });
    });

    it('should clone a compiler from the compiler pane dropdown', () => {
        // Set options on the first compiler
        findPane('Editor #').find('input.options').clear().type('-DVARIANT_A');
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'variant_a_func');

        // Clone from the compiler pane's "Add new" dropdown
        cy.get('[data-cy="new-compiler-dropdown-btn"]:visible').first().click();
        cy.get('[data-cy="new-add-compiler-btn"]:visible').first().click();

        // Should now have two compiler panes
        allCompilerTabs().should('have.length', 2);

        // The cloned compiler should also show variant_a_func (inherits options)
        allCompilerTabs()
            .last()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'variant_a_func');
                });
            });
    });

    it('should recompile all panes when source changes', () => {
        // Add second compiler with -DVARIANT_A
        findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
        cy.get('[data-cy="new-add-compiler-btn"]:visible').first().click();
        allCompilerTabs().should('have.length', 2);

        allCompilerTabs()
            .last()
            .then($tab => {
                compilerPaneFromTab($tab).find('input.options').clear().type('-DVARIANT_A');
            });

        // Now change source — both panes should recompile
        setMonacoEditorContent(
            [
                '#ifdef VARIANT_A',
                'int changed_a(int x) { return x + 100; }',
                '#else',
                'int changed_default(int x) { return x * 100; }',
                '#endif',
            ].join('\n'),
        );

        // First compiler (no flags) should show changed_default
        allCompilerTabs()
            .first()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'changed_default');
                });
            });

        // Second compiler (-DVARIANT_A) should show changed_a
        allCompilerTabs()
            .last()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'changed_a');
                });
            });
    });
});
