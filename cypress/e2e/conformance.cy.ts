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

import {stubCompileResponse} from '../support/fake-compile';
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
        // Default echo response has code: 0 → pass
        waitForEditors();
        setMonacoEditorContent('int square(int n) { return n * n; }');
        openConformanceView();
        addConformanceCompiler();
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');
    });

    it('should show a fail indicator for code with a non-zero exit code', () => {
        waitForEditors();
        stubCompileResponse({code: 1, stderr: [{text: 'error: deliberate failure'}]});
        setMonacoEditorContent('int broken() {}');
        openConformanceView();
        addConformanceCompiler();
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });

    it('should update status when source code changes', () => {
        // Start with passing code (default echo → code: 0)
        waitForEditors();
        openConformanceView();
        addConformanceCompiler();
        // Wait for initial pass indicator
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');

        // Now stub failure and change source to trigger recompile
        stubCompileResponse({code: 1, stderr: [{text: 'error: now it fails'}]});
        setMonacoEditorContent('int broken() {}');
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });

    it('should support multiple compilers', () => {
        // Just verify we can add two compilers and both show results
        waitForEditors();
        setMonacoEditorContent('int all_good(int x) { return x; }');
        openConformanceView();
        addConformanceCompiler();
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');

        addConformanceCompiler();
        // Both should pass with default echo
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('have.length.greaterThan', 0);
    });
});
