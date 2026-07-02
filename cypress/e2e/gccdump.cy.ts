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
    openGccDump,
    setupAndWaitForCompilation,
    visitPage,
} from '../support/utils';

// Anchor on the pane's own pass-picker rather than the GoldenLayout tab title: the title query
// (`span.lm_title:visible`) proved flaky right after a TomSelect dropdown interaction, whereas the
// `.gccdump-pass-picker` element is unique to this pane and always present in its content area.
function gccDumpPane() {
    return cy.get('.gccdump-pass-picker', {timeout: 10000}).closest('.lm_content');
}

beforeEach(visitPage);

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('GCC Tree/RTL dump', () => {
    it('should open a GCC dump pane from the compiler toolbar', () => {
        setupAndWaitForCompilation();
        openGccDump();
        gccDumpPane().should('exist');
    });

    it('should show a pass picker with available passes', () => {
        setupAndWaitForCompilation();
        openGccDump();
        cy.get('.gccdump-pass-picker + .ts-wrapper .ts-control', {timeout: 10000}).should('be.visible').click();
        cy.get('.ts-dropdown .option:visible', {timeout: 10000}).should('have.length.greaterThan', 0);
    });

    it('should display tree dump content when a pass is selected', () => {
        setupAndWaitForCompilation();
        openGccDump();
        cy.get('.gccdump-pass-picker + .ts-wrapper .ts-control', {timeout: 10000}).should('be.visible').click();
        cy.get('.ts-dropdown .option:visible', {timeout: 10000}).should('have.length.greaterThan', 0);
        // Pick a tree pass specifically: the first option is now an IPA summary (cgraph), whereas
        // this test is about the GIMPLE/tree body, which is what shows the `square` function.
        cy.contains('.ts-dropdown .option:visible', '(tree)').click();
        monacoEditorTextShouldContain(gccDumpPane().find('.monaco-editor'), 'square');
    });

    it('should switch passes without triggering a recompile', () => {
        // All pass dumps are shipped on the initial compile, so selecting a different pass is a
        // client-side lookup. Count compile requests and assert switching passes adds none.
        let compileCount = 0;
        cy.intercept('POST', '**/api/compiler/**/compile', req => {
            compileCount++;
            req.continue();
        }).as('compile');

        setupAndWaitForCompilation();
        openGccDump();

        // Wait for the picker to populate from the initial compile, then pick a first pass.
        cy.get('.gccdump-pass-picker + .ts-wrapper .ts-control', {timeout: 10000}).should('be.visible').click();
        cy.get('.ts-dropdown .option:visible', {timeout: 10000}).should('have.length.greaterThan', 1);
        cy.get('.ts-dropdown .option:visible').first().click();

        // Once settled, switch to a different pass and assert no new compile fired. (Content is
        // covered by the test above; here we only care that switching is a client-side lookup.)
        cy.then(() => {
            const before = compileCount;
            cy.get('.gccdump-pass-picker + .ts-wrapper .ts-control').click();
            cy.get('.ts-dropdown .option:visible').eq(1).click();
            // Give any (unwanted) recompile a chance to be issued before asserting.
            cy.wait(1500);
            cy.then(() => {
                expect(compileCount, 'compile requests during pass switch').to.equal(before);
            });
        });
    });
});
