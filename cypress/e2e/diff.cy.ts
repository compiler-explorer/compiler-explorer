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
    addCompilerFromEditor,
    assertNoConsoleOutput,
    findPane,
    lastCompilerContent,
    monacoEditorTextShouldContain,
    openDiffView,
    setMonacoEditorContent,
    visitPage,
    waitForEditors,
} from '../support/utils';

/**
 * Source code with conditional compilation for producing distinct outputs
 * in two compiler panes, making diff view show meaningful differences.
 * Uses different function names so we can assert the diff shows both variants.
 */
const DIFF_SOURCE = `\
#ifdef USE_ADD
int add_variant(int a, int b) { return a + b; }
#else
int mul_variant(int a, int b) { return a * b; }
#endif`;

beforeEach(visitPage);

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('Diff view', () => {
    it('should open a diff view pane from the Add menu', () => {
        waitForEditors();
        openDiffView();
    });

    it('should show diff content with two compiler panes', () => {
        waitForEditors();
        setMonacoEditorContent(DIFF_SOURCE);
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'mul_variant');

        // Add second compiler
        addCompilerFromEditor();
        cy.get('span.lm_title:visible').filter(':contains("Editor #")').should('have.length', 2);
        lastCompilerContent()
            .find('.monaco-editor .view-lines', {timeout: 10000})
            .should('contain.text', 'mul_variant');

        // Set -DUSE_ADD on the second compiler
        lastCompilerContent().find('input.options').clear().type('-DUSE_ADD');
        lastCompilerContent()
            .find('.monaco-editor .view-lines', {timeout: 10000})
            .should('contain.text', 'add_variant');

        // Open diff view — auto-selects the two compilers
        openDiffView();

        // Verify the diff shows content from at least one variant
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
