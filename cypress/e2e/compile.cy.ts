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
        // Wait for the compiler pane to have a Monaco editor with assembly output.
        // There should be at least 2 Monaco editors: source + compiler output.
        cy.get('.monaco-editor', {timeout: 10000}).should('have.length.at.least', 2);

        // The compiler output should contain x86 assembly instructions.
        // `imul` is what g++ emits for `num * num` at default optimisation.
        cy.get('.monaco-editor')
            .eq(1)
            .find('.view-lines', {timeout: 10000})
            .should('contain.text', 'square');
    });

    it('should recompile when the source code changes', () => {
        // Wait for initial compilation to finish
        cy.get('.monaco-editor', {timeout: 10000}).should('have.length.at.least', 2);
        cy.get('.monaco-editor').eq(1).find('.view-lines', {timeout: 10000}).should('contain.text', 'square');

        // Change the code to a different function
        setMonacoEditorContent('int cube(int n) { return n * n * n; }');

        // The compiler output should update to show the new function name
        cy.get('.monaco-editor')
            .eq(1)
            .find('.view-lines', {timeout: 10000})
            .should('contain.text', 'cube');
    });
});
