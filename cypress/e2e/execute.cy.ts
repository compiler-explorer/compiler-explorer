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
    compilerOutput,
    findPane,
    monacoEditorTextShouldContain,
    openExecutor,
    setMonacoEditorContent,
    setupAndWaitForCompilation,
    visitPage,
    waitForEditors,
} from '../support/utils';

function executorPane() {
    return findPane('Executor');
}

beforeEach(visitPage);

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('Executor', () => {
    it('should open an executor pane from the compiler toolbar', () => {
        setupAndWaitForCompilation();
        openExecutor();
        executorPane().should('exist');
    });

    it('should show program stdout', () => {
        waitForEditors();
        setMonacoEditorContent(`\
#include <cstdio>
int main() { printf("hello from cypress"); return 0; }`);
        monacoEditorTextShouldContain(compilerOutput(), 'main');
        openExecutor();
        executorPane().find('.execution-stdout', {timeout: 15000}).should('contain.text', 'hello from cypress');
    });

    it('should show non-zero exit code', () => {
        waitForEditors();
        setMonacoEditorContent('int main() { return 42; }');
        monacoEditorTextShouldContain(compilerOutput(), 'main');
        openExecutor();
        executorPane().find('.execution-output', {timeout: 15000}).should('contain.text', 'Program returned: 42');
    });

    it('should show stderr output', () => {
        waitForEditors();
        setMonacoEditorContent(`\
#include <cstdio>
int main() { fprintf(stderr, "error output"); return 0; }`);
        monacoEditorTextShouldContain(compilerOutput(), 'main');
        openExecutor();
        executorPane().find('.execution-output', {timeout: 15000}).should('contain.text', 'error output');
    });
});
