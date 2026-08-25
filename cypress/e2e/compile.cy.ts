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
    allCompilerTabs,
    assertNoConsoleOutput,
    compilerOutput,
    compilerPane,
    findPane,
    monacoEditorTextShouldContain,
    monacoEditorTextShouldNotContain,
    setMonacoEditorContent,
    setupAndWaitForCompilation,
    sourceEditor,
    visitPage,
    waitForEditors,
} from '../support/utils';

beforeEach(visitPage);

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('Basic compilation', () => {
    it('should compile the default code on load and show assembly', () => {
        waitForEditors();
        monacoEditorTextShouldContain(compilerOutput(), 'square');
    });

    it('should recompile when the source code changes', () => {
        setupAndWaitForCompilation();
        setMonacoEditorContent('int cube(int n) { return n * n * n; }');
        monacoEditorTextShouldContain(compilerOutput(), 'cube');
    });
});

describe('Source-assembly line linking', () => {
    beforeEach(() => {
        setMonacoEditorContent(`\
int square(int num) {
    return num * num;
}

int add(int a, int b) {
    return a + b;
}`);
        waitForEditors();
        monacoEditorTextShouldContain(compilerOutput(), 'square');
        monacoEditorTextShouldContain(compilerOutput(), 'add');
    });

    it('should highlight assembly lines when hovering over source code', () => {
        sourceEditor()
            .find('.view-line')
            .eq(1) // Line 2 (0-indexed): "return num * num;"
            .trigger('mousemove', {force: true});

        compilerOutput().find('.linked-code-decoration-margin', {timeout: 5000}).should('exist');
    });

    it('should highlight source lines when hovering over assembly', () => {
        compilerOutput().contains('.view-line', 'ret').first().trigger('mousemove', {force: true});
        sourceEditor().find('.linked-code-decoration-margin', {timeout: 5000}).should('exist');
    });
});

describe('Compiler options', () => {
    beforeEach(() => {
        setMonacoEditorContent(`\
#ifdef CYPRESS_TEST
int cypress_test_active(void) { return 1; }
#else
int cypress_test_inactive(void) { return 0; }
#endif`);
        waitForEditors();
        monacoEditorTextShouldContain(compilerOutput(), 'cypress_test_inactive');
    });

    it('should apply -D flag and recompile', () => {
        compilerPane().find('input.options').clear().type('-DCYPRESS_TEST');
        monacoEditorTextShouldContain(compilerOutput(), 'cypress_test_active');
        monacoEditorTextShouldNotContain(compilerOutput(), 'cypress_test_inactive');
    });
});

describe('Output filters', () => {
    beforeEach(() => {
        setupAndWaitForCompilation();
    });

    it('should show directives when directive filter is toggled off', () => {
        compilerOutput().find('.view-lines').should('not.contain.text', '.cfi_startproc');

        compilerPane().find('button[title="Compiler output filters"]').click();
        cy.get('button[data-bind="directives"]:visible').first().click();

        compilerOutput().find('.view-lines', {timeout: 10000}).should('contain.text', '.cfi_startproc');
    });

    it('should show comment-only lines when comment filter is toggled off', () => {
        compilerPane().find('button[title="Compiler output filters"]').click();
        cy.get('button[data-bind="commentOnly"]:visible').first().click();

        monacoEditorTextShouldContain(compilerOutput(), 'GNU C++');
    });
});

describe('Compilation errors', () => {
    it('should display compilation failure message for invalid code', () => {
        waitForEditors();
        setMonacoEditorContent('int main() { this is not valid c++; }');
        monacoEditorTextShouldContain(compilerOutput(), 'Compilation failed');
    });
});

describe('Output pane', () => {
    it('should show clean output for valid code', () => {
        setupAndWaitForCompilation();
        cy.get('[data-cy="new-output-pane-btn"]:visible').first().click();
        findPane('Output').find('.content', {timeout: 5000}).should('contain.text', 'Compiler returned: 0');
    });

    it('should show error details for invalid code', () => {
        setupAndWaitForCompilation();
        cy.get('[data-cy="new-output-pane-btn"]:visible').first().click();
        findPane('Output').find('.content', {timeout: 5000}).should('contain.text', 'Compiler returned: 0');

        setMonacoEditorContent('int main() { this is not valid c++; }');

        findPane('Output').find('.content', {timeout: 10000}).should('contain.text', 'error:');
        findPane('Output').find('.content', {timeout: 10000}).should('not.contain.text', 'Compiler returned: 0');
    });

    it('should show stderr with source location for errors', () => {
        setMonacoEditorContent('int main() { return missing_variable; }');
        waitForEditors();
        cy.get('[data-cy="new-output-pane-btn"]:visible').first().click();
        findPane('Output').find('.content', {timeout: 10000}).should('contain.text', 'missing_variable');
    });
});

describe('Editor interactions', () => {
    it('should add a second source editor', () => {
        setupAndWaitForCompilation();
        findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
        cy.get('[data-cy="new-add-editor-btn"]:visible').first().click();
        cy.get('span.lm_title:visible').filter(':contains("source")').should('have.length', 2);
    });

    it('should add a second compiler', () => {
        setupAndWaitForCompilation();
        addCompilerFromEditor();
        allCompilerTabs().should('have.length', 2);
    });

    it('should show assembly in both compilers for the same source', () => {
        setupAndWaitForCompilation();
        addCompilerFromEditor();
        allCompilerTabs().should('have.length', 2);

        allCompilerTabs().each($title => {
            cy.wrap($title)
                .closest('.lm_item.lm_stack')
                .find('.lm_content .monaco-editor .view-lines', {timeout: 10000})
                .should('contain.text', 'square');
        });
    });
});
