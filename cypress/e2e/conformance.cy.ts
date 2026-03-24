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

    it('should show a fail indicator for code with static_assert failure', () => {
        waitForEditors();
        setMonacoEditorContent('static_assert(false, "deliberate failure");');
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

        setMonacoEditorContent('static_assert(false, "now it fails");');
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });

    it('should support multiple compilers with different options', () => {
        waitForEditors();
        setMonacoEditorContent(`\
#ifdef FAIL_ME
static_assert(false, "conditional failure");
#else
int all_good(int x) { return x; }
#endif`);

        openConformanceView();

        // First compiler — should pass (no -D flag)
        addConformanceCompiler();
        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');

        // Second compiler with -DFAIL_ME — should fail
        addConformanceCompiler();
        conformancePane().find('.conformance-options').last().clear().type('-DFAIL_ME');

        conformancePane().find('.fa-check-circle', {timeout: 10000}).should('exist');
        conformancePane().find('.fa-times-circle', {timeout: 10000}).should('exist');
    });
});
