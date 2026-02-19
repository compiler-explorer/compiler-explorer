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
    addConformanceCompiler,
    assertNoConsoleOutput,
    conformancePane,
    openConformanceView,
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

describe('Conformance view', () => {
    it('should open a conformance view pane', () => {
        waitForEditors();
        openConformanceView();
        cy.get('.conformance-wrapper:visible').should('exist');
    });

    it('should show a pass indicator for code that compiles cleanly', () => {
        waitForEditors();
        setMonacoEditorContent('int square(int n) { return n * n; }');
        openConformanceView();
        addConformanceCompiler();
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');
    });

    it('should show a fail indicator for code with a non-zero exit code', () => {
        waitForEditors();
        setMonacoEditorContent(`\
// FAKE: exitcode 1
// FAKE: stderr error: deliberate failure
int broken() {}`);
        openConformanceView();
        addConformanceCompiler();
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });

    it('should update status when source code changes', () => {
        waitForEditors();
        setMonacoEditorContent('int valid_func(int x) { return x; }');
        openConformanceView();
        addConformanceCompiler();
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');

        setMonacoEditorContent(`\
// FAKE: exitcode 1
// FAKE: stderr error: now it fails
int broken() {}`);
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });

    it('should support multiple compilers with different options', () => {
        waitForEditors();
        setMonacoEditorContent('int all_good(int x) { return x; }');

        openConformanceView();

        // First compiler — should pass (no flags)
        addConformanceCompiler();
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');

        // Second compiler with --fake-exitcode=1 — should fail
        addConformanceCompiler();
        conformancePane().find('.conformance-options').last().clear().type('--fake-exitcode=1');

        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });
});
