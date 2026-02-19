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
        sourceEditor().find('.view-line').eq(1).trigger('mousemove', {force: true});

        compilerOutput().find('.linked-code-decoration-margin', {timeout: 5000}).should('exist');
    });

    it('should highlight source lines when hovering over assembly', () => {
        compilerOutput().find('.view-line').eq(1).trigger('mousemove', {force: true});
        sourceEditor().find('.linked-code-decoration-margin', {timeout: 5000}).should('exist');
    });
});

describe('Compiler options', () => {
    it('should apply options and recompile', () => {
        setupAndWaitForCompilation();
        compilerPane().find('input.options').clear().type('-O2 -Wall');
        monacoEditorTextShouldContain(compilerOutput(), '-O2 -Wall');
    });
});

describe('Compilation errors', () => {
    it('should display compilation failure in output pane', () => {
        waitForEditors();
        setMonacoEditorContent(`\
// FAKE: exitcode 1
// FAKE: stderr error: expected ';' before '}' token
int main() { broken }`);
        cy.get('[data-cy="new-output-pane-btn"]:visible').first().click();
        findPane('Output').find('.content', {timeout: 10000}).should('contain.text', "expected ';'");
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

        setMonacoEditorContent(`\
// FAKE: exitcode 1
// FAKE: stderr error: use of undeclared identifier 'missing_variable'
int main() { return missing_variable; }`);

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
