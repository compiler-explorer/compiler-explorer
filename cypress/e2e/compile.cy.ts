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
    monacoEditorTextShouldNotContain,
    setMonacoEditorContent,
    stubConsoleOutput,
} from '../support/utils';

/** Helper to get the compiler output editor (second Monaco editor on the page). */
function compilerOutput() {
    return cy.get('.monaco-editor').eq(1);
}

/** Helper to get the source editor (first Monaco editor on the page). */
function sourceEditor() {
    return cy.get('.monaco-editor').first();
}

/** Wait for the page to load with both source and compiler editors visible. */
function waitForEditors() {
    cy.get('.monaco-editor', {timeout: 10000}).should('have.length.at.least', 2);
}

/** Common setup: visit page, stub console, wait for editors. */
function setupAndWaitForCompilation() {
    waitForEditors();
    monacoEditorTextShouldContain(compilerOutput(), 'square');
}

// All tests share the same visit + console stub/assert pattern.
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
        // Use two distinct functions so we can verify correct line mapping
        setMonacoEditorContent(
            [
                'int square(int num) {',
                '    return num * num;',
                '}',
                '',
                'int add(int a, int b) {',
                '    return a + b;',
                '}',
            ].join('\n'),
        );
        waitForEditors();
        monacoEditorTextShouldContain(compilerOutput(), 'square');
        monacoEditorTextShouldContain(compilerOutput(), 'add');
    });

    it('should highlight assembly lines when hovering over source code', () => {
        // Hover over a source line containing code (line 2: "return num * num;").
        // Monaco picks up mousemove events on .view-line elements within the editor.
        sourceEditor()
            .find('.view-line')
            .eq(1) // Line 2 (0-indexed)
            .trigger('mousemove', {force: true});

        // The compiler pane should now show linked-code decorations
        compilerOutput().find('.linked-code-decoration-margin', {timeout: 5000}).should('exist');
    });

    it('should highlight source lines when hovering over assembly', () => {
        // Hover over an assembly line in the compiler output.
        compilerOutput()
            .find('.view-line')
            .eq(2) // An instruction line
            .trigger('mousemove', {force: true});

        // The source editor should now show linked-code decorations
        sourceEditor().find('.linked-code-decoration-margin', {timeout: 5000}).should('exist');
    });
});

describe('Compiler options', () => {
    beforeEach(() => {
        // Use code with a preprocessor conditional so we can verify -D takes effect
        setMonacoEditorContent(
            [
                '#ifdef CYPRESS_TEST',
                'int cypress_test_active(void) { return 1; }',
                '#else',
                'int cypress_test_inactive(void) { return 0; }',
                '#endif',
            ].join('\n'),
        );
        waitForEditors();
        monacoEditorTextShouldContain(compilerOutput(), 'cypress_test_inactive');
    });

    it('should apply -D flag and recompile', () => {
        cy.get('input.options').first().clear().type('-DCYPRESS_TEST{enter}');

        monacoEditorTextShouldContain(compilerOutput(), 'cypress_test_active');
        monacoEditorTextShouldNotContain(compilerOutput(), 'cypress_test_inactive');
    });
});

describe('Output filters', () => {
    beforeEach(() => {
        setupAndWaitForCompilation();
    });

    it('should show directives when directive filter is toggled off', () => {
        // By default, directives are filtered out.
        compilerOutput().find('.view-lines').should('not.contain.text', '.cfi_startproc');

        // Toggle directives filter off
        cy.get('button[title="Compiler output filters"]').first().click();
        cy.get('button[data-bind="directives"]').first().click();

        // Now directives should appear
        compilerOutput().find('.view-lines', {timeout: 10000}).should('contain.text', '.cfi_startproc');
    });

    it('should show comment-only lines when comment filter is toggled off', () => {
        // Toggle the comment filter off
        cy.get('button[title="Compiler output filters"]').first().click();
        cy.get('button[data-bind="commentOnly"]').first().click();

        // GCC emits comment lines like "# GNU C++17..." at the top of assembly output.
        monacoEditorTextShouldContain(compilerOutput(), 'GNU C++');
    });
});

describe('Compilation errors', () => {
    it('should display compilation failure message for invalid code', () => {
        setMonacoEditorContent('int main() { this is not valid c++; }');

        monacoEditorTextShouldContain(compilerOutput(), 'Compilation failed');
    });
});
