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
    compilerPane,
    monacoEditorTextShouldContain,
    monacoEditorTextShouldNotContain,
    setMonacoEditorContent,
    sourceEditor,
    stubConsoleOutput,
    waitForEditors,
} from '../support/utils';

beforeEach(() => {
    cy.visit('/', {
        onBeforeLoad: win => {
            stubConsoleOutput(win);
            // Disable auto-compile so we can test that Ctrl+Enter explicitly triggers it
            win.localStorage.setItem('settings', JSON.stringify({compileOnChange: false}));
        },
    });
});

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('Keyboard shortcuts', () => {
    it('should recompile with Ctrl+Enter when compileOnChange is disabled', () => {
        waitForEditors();
        setMonacoEditorContent(`\
#ifdef SHORTCUT_TEST
int shortcut_active(void) { return 1; }
#else
int shortcut_inactive(void) { return 0; }
#endif`);

        // With compileOnChange off, changing options should NOT recompile
        compilerPane().find('input.options').clear().type('-DSHORTCUT_TEST');

        // Output should still show the old compilation result (no -D flag)
        monacoEditorTextShouldNotContain(compilerOutput(), 'shortcut_active');

        // Ctrl+Enter should trigger recompilation
        sourceEditor().find('textarea').type('{ctrl}{enter}', {force: true});

        monacoEditorTextShouldContain(compilerOutput(), 'shortcut_active');
    });
});
