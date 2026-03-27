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

const SOURCE = 'int func(int x) { return x * 2; }';

beforeEach(visitPage);

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('Multi-compiler panes', () => {
    beforeEach(() => {
        waitForEditors();
        setMonacoEditorContent(SOURCE);
    });

    it('should add a second compiler pane from the editor dropdown', () => {
        addCompilerFromEditor();
        allCompilerTabs().should('have.length', 2);
    });

    it('should produce different output with different options', () => {
        // First compiler: no options → echoes source
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'func');

        // Add second compiler with -O2 → output includes '; Options: -O2'
        addCompilerFromEditor();
        allCompilerTabs().should('have.length', 2);

        allCompilerTabs()
            .last()
            .then($tab => {
                compilerPaneFromTab($tab).find('input.options').clear().type('-O2');
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), '-O2');
                });
            });

        // First compiler should NOT show -O2
        allCompilerTabs()
            .first()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'func');
                });
            });
    });

    it('should clone a compiler and inherit its options', () => {
        // Set options on the first compiler
        findPane('Editor #').find('input.options').clear().type('-O3');
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), '-O3');

        // Clone from the compiler pane's dropdown
        addCompilerFromCompilerPane();
        allCompilerTabs().should('have.length', 2);

        // The cloned compiler should also show -O3
        allCompilerTabs()
            .last()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), '-O3');
                });
            });
    });

    it('should recompile all panes when source changes', () => {
        addCompilerFromEditor();
        allCompilerTabs().should('have.length', 2);

        allCompilerTabs()
            .last()
            .then($tab => {
                compilerPaneFromTab($tab).find('input.options').clear().type('-O2');
            });

        // Change source — both panes should recompile with new source echoed
        setMonacoEditorContent('int changed(int x) { return x + 100; }');

        allCompilerTabs()
            .first()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'changed');
                });
            });

        allCompilerTabs()
            .last()
            .then($tab => {
                compilerEditorFromTab($tab).then($editor => {
                    monacoEditorTextShouldContain(cy.wrap($editor), 'changed');
                });
            });
    });
});
