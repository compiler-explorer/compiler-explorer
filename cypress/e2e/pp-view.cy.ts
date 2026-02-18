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
    findPane,
    monacoEditorTextShouldContain,
    monacoEditorTextShouldNotContain,
    openPreprocessor,
    setMonacoEditorContent,
    setupAndWaitForCompilation,
    visitPage,
    waitForEditors,
} from '../support/utils';

function ppPane() {
    return findPane('Preprocessor');
}

beforeEach(visitPage);

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('Preprocessor output', () => {
    it('should open a preprocessor pane from the compiler toolbar', () => {
        setupAndWaitForCompilation();
        openPreprocessor();
        ppPane().should('exist');
    });

    it('should show expanded #define substitution', () => {
        setMonacoEditorContent(`\
#define MAGIC 42
int get_magic() { return MAGIC; }`);
        waitForEditors();
        openPreprocessor();
        monacoEditorTextShouldContain(ppPane().find('.monaco-editor'), '42');
    });

    it('should expand macros from #include', () => {
        setMonacoEditorContent(`\
#include <climits>
int max_int() { return INT_MAX; }`);
        waitForEditors();
        openPreprocessor();
        monacoEditorTextShouldContain(ppPane().find('.monaco-editor'), 'max_int');
        monacoEditorTextShouldNotContain(ppPane().find('.monaco-editor'), 'INT_MAX');
    });

    it('should update when source changes', () => {
        setMonacoEditorContent(`\
#define VALUE_A 100
int a() { return VALUE_A; }`);
        waitForEditors();
        openPreprocessor();
        monacoEditorTextShouldContain(ppPane().find('.monaco-editor'), '100');

        setMonacoEditorContent(`\
#define VALUE_B 999
int b() { return VALUE_B; }`);
        monacoEditorTextShouldContain(ppPane().find('.monaco-editor'), '999');
    });
});
