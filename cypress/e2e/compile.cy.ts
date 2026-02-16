// Copyright (c) 2025, Compiler Explorer Authors
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

describe('Basic compilation', () => {
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

    it('should compile the default code on load and show assembly', () => {
        // The default code (square function) compiles automatically on page load.
        // There should be at least 2 Monaco editors: source + compiler output.
        cy.get('.monaco-editor', {timeout: 10000}).should('have.length.at.least', 2);

        // The compiler output should contain the function name in assembly labels.
        cy.get('.monaco-editor').eq(1).find('.view-lines', {timeout: 10000}).should('contain.text', 'square');
    });

    it('should recompile when the source code changes', () => {
        // Wait for initial compilation to finish
        cy.get('.monaco-editor', {timeout: 10000}).should('have.length.at.least', 2);
        cy.get('.monaco-editor').eq(1).find('.view-lines', {timeout: 10000}).should('contain.text', 'square');

        // Change the code to a different function
        setMonacoEditorContent('int cube(int n) { return n * n * n; }');

        // The compiler output should update to show the new function name
        cy.get('.monaco-editor').eq(1).find('.view-lines', {timeout: 10000}).should('contain.text', 'cube');
    });
});

describe('Source-assembly line linking', () => {
    beforeEach(() => {
        cy.visit('/', {
            onBeforeLoad: win => {
                stubConsoleOutput(win);
            },
        });
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
        // Wait for compilation to produce output containing both functions
        cy.get('.monaco-editor', {timeout: 10000}).should('have.length.at.least', 2);
        cy.get('.monaco-editor').eq(1).find('.view-lines', {timeout: 10000}).should('contain.text', 'square');
        cy.get('.monaco-editor').eq(1).find('.view-lines').should('contain.text', 'add');
    });

    afterEach(() => {
        return cy.window().then(_win => {
            assertNoConsoleOutput();
        });
    });

    it('should highlight assembly lines when hovering over source code', () => {
        // Hover over a source line containing code (line 2: "return num * num;").
        // Monaco picks up mousemove events on .view-line elements within the editor.
        cy.get('.monaco-editor')
            .first()
            .find('.view-line')
            .eq(1) // Line 2 (0-indexed)
            .trigger('mousemove', {force: true});

        // The compiler pane should now show linked-code decorations
        cy.get('.monaco-editor').eq(1).find('.linked-code-decoration-margin', {timeout: 5000}).should('exist');
    });

    it('should highlight source lines when hovering over assembly', () => {
        // Hover over an assembly line in the compiler output.
        // Skip line 0 (likely a label) and hover over an instruction line.
        cy.get('.monaco-editor')
            .eq(1)
            .find('.view-line')
            .eq(2) // An instruction line
            .trigger('mousemove', {force: true});

        // The source editor should now show linked-code decorations
        cy.get('.monaco-editor').first().find('.linked-code-decoration-margin', {timeout: 5000}).should('exist');
    });
});

describe('Compiler options', () => {
    beforeEach(() => {
        cy.visit('/', {
            onBeforeLoad: win => {
                stubConsoleOutput(win);
            },
        });
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
        // Wait for initial compilation (without -D, should see the #else branch)
        cy.get('.monaco-editor', {timeout: 10000}).should('have.length.at.least', 2);
        cy.get('.monaco-editor')
            .eq(1)
            .find('.view-lines', {timeout: 10000})
            .should('contain.text', 'cypress_test_inactive');
    });

    afterEach(() => {
        return cy.window().then(_win => {
            assertNoConsoleOutput();
        });
    });

    it('should apply -D flag and recompile', () => {
        // Type -DCYPRESS_TEST into the compiler options field
        cy.get('input.options').first().clear().type('-DCYPRESS_TEST{enter}');

        // Now the #ifdef branch should be active
        cy.get('.monaco-editor')
            .eq(1)
            .find('.view-lines', {timeout: 10000})
            .should('contain.text', 'cypress_test_active');
        // And the #else branch should be gone
        cy.get('.monaco-editor').eq(1).find('.view-lines').should('not.contain.text', 'cypress_test_inactive');
    });
});

describe('Output filters', () => {
    beforeEach(() => {
        cy.visit('/', {
            onBeforeLoad: win => {
                stubConsoleOutput(win);
            },
        });
        // Wait for compilation of default code
        cy.get('.monaco-editor', {timeout: 10000}).should('have.length.at.least', 2);
        cy.get('.monaco-editor').eq(1).find('.view-lines', {timeout: 10000}).should('contain.text', 'square');
    });

    afterEach(() => {
        return cy.window().then(_win => {
            assertNoConsoleOutput();
        });
    });

    it('should show directives when directive filter is toggled off', () => {
        // By default, directives are filtered out. The assembly should NOT contain directives.
        cy.get('.monaco-editor').eq(1).find('.view-lines').should('not.contain.text', '.cfi_startproc');

        // Open the filter dropdown and toggle directives off
        cy.get('button[title="Compiler output filters"]').first().click();
        cy.get('button[data-bind="directives"]').first().click();

        // Now directives should appear in the output
        cy.get('.monaco-editor').eq(1).find('.view-lines', {timeout: 10000}).should('contain.text', '.cfi_startproc');
    });

    it('should show comment-only lines when comment filter is toggled off', () => {
        // By default, comment-only lines are filtered out.
        // Toggle the comment filter off
        cy.get('button[title="Compiler output filters"]').first().click();
        cy.get('button[data-bind="commentOnly"]').first().click();

        // GCC emits comment lines like "# GNU C++17..." at the top of assembly output.
        // With comments filtered (default), these are hidden. With filter off, they appear.
        // Note: Monaco uses non-breaking spaces (U+00A0) in rendered text, so we normalise
        // before comparing by replacing all \u00a0 with regular spaces.
        cy.get('.monaco-editor')
            .eq(1)
            .find('.view-lines', {timeout: 10000})
            .should($el => {
                const text = $el.text().replaceAll('\u00a0', ' ');
                expect(text).to.include('GNU C++');
            });
    });
});

describe('Compilation errors', () => {
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

    it('should display compilation failure message for invalid code', () => {
        setMonacoEditorContent('int main() { this is not valid c++; }');

        // The compiler pane shows "<Compilation failed>" when compilation fails.
        // Monaco uses non-breaking spaces (U+00A0), so we normalise before checking.
        cy.get('.monaco-editor')
            .eq(1)
            .find('.view-lines', {timeout: 10000})
            .should($el => {
                const text = $el.text().replaceAll('\u00a0', ' ');
                expect(text).to.include('Compilation failed');
            });
    });
});
