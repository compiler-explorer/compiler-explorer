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
    addCompilerFromCompilerPane,
    addCompilerFromEditor,
    allCompilerTabs,
    assertNoConsoleOutput,
    compilerEditorFromTab,
    compilerPaneFromTab,
    findPane,
    monacoEditorTextShouldContain,
    setMonacoEditorContent,
    visitPage,
    waitForEditors,
} from '../support/utils';

/**
 * Source code using #ifdef to produce different function names based on -D flags.
 * This lets us verify that two compiler panes with different options produce
 * different assembly output using only the single available compiler (gdefault).
 */
const CONDITIONAL_SOURCE = `\
#ifdef VARIANT_A
int variant_a_func(int x) { return x + 1; }
#elif defined(VARIANT_B)
int variant_b_func(int x) { return x - 1; }
#else
int default_func(int x) { return x * 2; }
#endif`;

beforeEach(visitPage);

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
        addCompilerFromEditor();
        allCompilerTabs().should('have.length', 2);
    });

    it('should produce different output with different -D flags', () => {
        // First compiler: default (no flags) → should show default_func
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'default_func');

        // Add second compiler with -DVARIANT_A
        addCompilerFromEditor();
        allCompilerTabs().should('have.length', 2);

        allCompilerTabs()
            .last()
            .then($tab => {
                compilerPaneFromTab($tab).find('input.options').clear().type('-DVARIANT_A');
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'variant_a_func');
                });
            });

        // First compiler should still show default_func
        allCompilerTabs()
            .first()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'default_func');
                });
            });
    });

    it('should clone a compiler and inherit its options', () => {
        // Set options on the first compiler
        findPane('Editor #').find('input.options').clear().type('-DVARIANT_A');
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'variant_a_func');

        // Clone from the compiler pane's dropdown
        addCompilerFromCompilerPane();
        allCompilerTabs().should('have.length', 2);

        // The cloned compiler should also show variant_a_func
        allCompilerTabs()
            .last()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'variant_a_func');
                });
            });
    });

    it('should recompile all panes when source changes', () => {
        addCompilerFromEditor();
        allCompilerTabs().should('have.length', 2);

        allCompilerTabs()
            .last()
            .then($tab => {
                compilerPaneFromTab($tab).find('input.options').clear().type('-DVARIANT_A');
            });

        // Change source — both panes should recompile
        setMonacoEditorContent(`\
#ifdef VARIANT_A
int changed_a(int x) { return x + 100; }
#else
int changed_default(int x) { return x * 100; }
#endif`);

        allCompilerTabs()
            .first()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'changed_default');
                });
            });

        allCompilerTabs()
            .last()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'changed_a');
                });
            });
    });
});
