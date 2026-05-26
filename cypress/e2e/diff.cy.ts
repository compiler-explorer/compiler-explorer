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
        setMonacoEditorContent('int func(int x) { return x; }');
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'func');

        // Add second compiler with different options so output differs
        addCompilerFromEditor();
        cy.get('span.lm_title:visible').filter(':contains("Editor #")').should('have.length', 2);
        lastCompilerContent().find('input.options').clear().type('-O2');
        lastCompilerContent().find('.monaco-editor .view-lines', {timeout: 10000}).should('contain.text', '-O2');

        // Open diff view — auto-selects the two compilers
        openDiffView();

        // Verify the diff shows content
        findPane('Diff')
            .find('.view-lines', {timeout: 10000})
            .should($el => {
                const text = $el.text().replaceAll('\u00a0', ' ');
                expect(text).to.include('func');
            });
    });
});
