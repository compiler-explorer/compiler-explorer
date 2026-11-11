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

// These tests cover the places where Compiler Explorer reaches into Monaco (or monaco-vim) internals that are not part
// of Monaco's documented API. A Monaco upgrade that moves or renames any of them should fail here, not in production.

import {serialiseState} from '../../shared/url-serialization.js';
import {
    compilerOutput,
    setMonacoEditorContent,
    setupAndWaitForCompilation,
    sourceEditor,
    stubConsoleOutput,
    visitPage,
    waitForEditors,
} from '../support/utils';

function sourceModelValue(): Cypress.Chainable<string> {
    return cy.window().then((win: Cypress.AUTWindow) => win.monaco.editor.getEditors()[0].getModel()!.getValue());
}

function sourceLine(lineNumber: number): Cypress.Chainable<string> {
    return cy
        .window()
        .then((win: Cypress.AUTWindow) => win.monaco.editor.getEditors()[0].getModel()!.getLineContent(lineNumber));
}

function sourceCursorLine(): Cypress.Chainable<number> {
    return cy.window().then((win: Cypress.AUTWindow) => win.monaco.editor.getEditors()[0].getPosition()!.lineNumber);
}

describe('Monaco language workers', () => {
    beforeEach(visitPage);

    it('runs the TypeScript worker and reports diagnostics', () => {
        waitForEditors();
        let resource: ReturnType<Cypress.AUTWindow['monaco']['Uri']['parse']>;
        cy.window().then((win: Cypress.AUTWindow) => {
            resource = win.monaco.Uri.parse('inmemory://worker-test.ts');
            win.monaco.editor.createModel('const x: number = "not a number";', 'typescript', resource);
        });
        cy.window({timeout: 30_000}).should((win: Cypress.AUTWindow) => {
            const markers = win.monaco.editor.getModelMarkers({resource});
            expect(markers.map(marker => marker.message).join('\n')).to.contain('not assignable');
        });
    });
});

describe('Vim mode', () => {
    beforeEach(() => {
        cy.visit('/', {
            onBeforeLoad: win => {
                win.localStorage.setItem('settings', JSON.stringify({useVim: true}));
                stubConsoleOutput(win);
            },
        });
        waitForEditors();
        setMonacoEditorContent('first\nsecond\nthird');
        sourceEditor().click();
        // Type into the focused input: typing into the editor element would click it again and move the cursor
        cy.focused().type('gg');
    });

    it('shows the vim status bar and enters insert mode on "i"', () => {
        cy.get('.v-status').should('be.visible');
        cy.focused().type('i');
        cy.get('.v-status').should('contain.text', 'INSERT');
    });

    it('leaves insert mode on Escape', () => {
        cy.focused().type('i');
        cy.get('.v-status').should('contain.text', 'INSERT');
        cy.focused().type('{esc}');
        cy.get('.v-status').should('not.contain.text', 'INSERT');
    });

    it('moves the cursor with G and gg', () => {
        cy.window().then((win: Cypress.AUTWindow) => {
            win.monaco.editor.getEditors()[0].setPosition({lineNumber: 2, column: 1});
        });
        cy.focused().type('G');
        sourceCursorLine().should('equal', 3);
        cy.focused().type('gg');
        sourceCursorLine().should('equal', 1);
    });

    it('runs normal mode commands against the model', () => {
        cy.focused().type('dd');
        sourceModelValue().should('equal', 'second\nthird');
    });

    it("indents with >> (monaco-vim drives Monaco's internal ShiftCommand)", () => {
        cy.focused().type('>>');
        sourceModelValue().should('match', /^\s+first\n/);
    });

    // Lines already on a tab stop take a simpler path through the shift command than lines that are not
    it('indents a line that is not on a tab stop with >> to the next one', () => {
        setMonacoEditorContent('a\n  b\nc');
        cy.focused().type('ggj>>');
        sourceLine(2).should('equal', '    b');
    });

    it('unindents a line that is not on a tab stop with << to the previous one', () => {
        setMonacoEditorContent('a\n      b\nc');
        cy.focused().type('ggj<<');
        sourceLine(2).should('equal', '    b');
    });
});

describe('Print view', () => {
    beforeEach(visitPage);

    it('extracts the Monaco theme CSS when printing', () => {
        setupAndWaitForCompilation();
        cy.window().then((win: Cypress.AUTWindow) => {
            win.dispatchEvent(new win.Event('beforeprint'));
        });
        cy.get('#printview style').invoke('text').should('contain', '#printview .mtk1');
    });
});

describe('Compiler pane context menu', () => {
    beforeEach(visitPage);

    it('opens through the patched context menu handler and offers the asm documentation action', () => {
        setupAndWaitForCompilation();
        compilerOutput().find('.view-line span').contains(/^ret$/).first().rightclick();
        // Monaco renders its context menu inside a shadow root
        cy.contains('.monaco-menu .action-label', 'View assembly documentation', {includeShadowDom: true})
            .closest('.action-item')
            .should('not.have.class', 'disabled');
    });

    it('does not offer the asm documentation action when the click is not on an instruction', () => {
        setupAndWaitForCompilation();
        // The context key defaults to true, so this only passes if the patched handler updates it for the click
        compilerOutput()
            .find('.view-line span')
            .contains(/^square/)
            .first()
            .rightclick();
        cy.contains('.monaco-menu .action-label', 'Copy', {includeShadowDom: true}).should('exist');
        cy.contains('.monaco-menu .action-label', 'View assembly documentation', {includeShadowDom: true}).should(
            'not.exist',
        );
    });
});

describe('Hidden tab editor click hit-testing', () => {
    it('positions the cursor in the middle of line 1 after switching from a hidden tab', () => {
        const hiddenLine1 = 'int square(int num) {';
        const state = {
            version: 4,
            content: [
                {
                    type: 'row',
                    content: [
                        {
                            type: 'stack',
                            activeItemIndex: 0,
                            content: [
                                {
                                    type: 'component',
                                    componentName: 'codeEditor',
                                    componentState: {id: 1, lang: 'c++', source: '// visible tab\n'},
                                },
                                {
                                    type: 'component',
                                    componentName: 'codeEditor',
                                    componentState: {
                                        id: 2,
                                        lang: 'c++',
                                        source: `${hiddenLine1}\n    return num * num;\n}`,
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        };
        const linuxUA =
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36';

        cy.visit(`/#${serialiseState(state)}`, {
            onBeforeLoad(win) {
                stubConsoleOutput(win);
                // Force Monaco's RenderedViewLine path on every OS so hidden DOM measurements cache _cachedWidth = 0
                Object.defineProperty(win.navigator, 'userAgent', {
                    value: linuxUA,
                    configurable: true,
                });
            },
        });

        cy.contains('span.lm_title:visible', 'C++ source #1', {timeout: 30_000}).should('be.visible');

        // Wait for Monaco's 200ms _asyncUpdateLineWidths timer to fire while the second tab is still hidden
        cy.window().should((win: Cypress.AUTWindow) => {
            const hiddenEditor = win.monaco.editor
                .getEditors()
                .find(ed => ed.getModel()?.getLineContent(1) === hiddenLine1) as any;
            const viewLines = hiddenEditor?._modelData?.view?._viewLines;
            expect(viewLines).to.exist;
            expect(viewLines.onFlushed).to.be.a('function');
            expect(viewLines._maxLineWidth).to.be.a('number');
            expect(viewLines._asyncUpdateLineWidths.isScheduled()).to.be.false;
        });

        cy.contains('span.lm_title:visible', 'C++ source #2').click();
        cy.contains('.monaco-editor:visible .view-lines', 'return num * num').should('be.visible');

        cy.get('.monaco-editor:visible .view-line')
            .first()
            .then($line1 => {
                const rect = $line1[0].getBoundingClientRect();
                const $editor = $line1.closest('.monaco-editor');
                const editorRect = $editor[0].getBoundingClientRect();
                // Click inside the middle of line 1 rather than past the end of the line
                const clickX = Math.round(rect.left + 80 - editorRect.left);
                const clickY = Math.round(rect.top + rect.height / 2 - editorRect.top);
                cy.wrap($editor).click(clickX, clickY);
            });

        cy.window().then((win: Cypress.AUTWindow) => {
            const visibleEditor = win.monaco.editor
                .getEditors()
                .find(ed => ed.getModel()?.getLineContent(1) === hiddenLine1)!;
            const pos = visibleEditor.getPosition()!;
            const maxColumn = visibleEditor.getModel()!.getLineMaxColumn(1);
            expect(pos.lineNumber).to.equal(1);
            expect(pos.column).to.be.greaterThan(1).and.lessThan(maxColumn);
        });
    });
});
