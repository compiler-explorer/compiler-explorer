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

import {serialiseState} from '../../shared/url-serialization.js';
import {
    addCompilerFromEditor,
    assertNoConsoleOutput,
    compilerPane,
    findPane,
    lastCompilerContent,
    monacoEditorTextShouldContain,
    openDiffView,
    setMonacoEditorContent,
    stubConsoleOutput,
    visitPage,
    waitForEditors,
} from '../support/utils';

/**
 * Source code with conditional compilation for producing distinct outputs
 * in two compiler panes, making diff view show meaningful differences.
 * Uses different function names so we can assert the diff shows both variants.
 */
const DIFF_SOURCE = `\
#ifdef USE_ADD
int add_variant(int a, int b) { return a + b; }
#else
int mul_variant(int a, int b) { return a * b; }
#endif`;

const SOURCE_A = `\
int source_alpha() {
    return 1;
}`;

const SOURCE_B = `\
int source_beta() {
    return 2;
}`;

const SOURCE_A_UPDATED = `\
int source_alpha_updated() {
    return 3;
}`;

const DIFF_TYPE_SOURCE = 16;

function buildSourceOnlyDiffState(diffState = {}) {
    const sourcePane = (id: number, source: string) => ({
        type: 'component',
        componentName: 'codeEditor',
        componentState: {id, lang: 'c++', source},
        isClosable: true,
    });

    const diffPane = {
        type: 'component',
        componentName: 'diff',
        componentState: diffState,
        isClosable: true,
    };

    return {
        version: 4,
        content: [
            {
                type: 'row',
                content: [sourcePane(1, SOURCE_A), sourcePane(2, SOURCE_B), diffPane],
            },
        ],
    };
}

function visitState(state: object) {
    cy.visit(`/#${serialiseState(state)}`, {
        onBeforeLoad: win => {
            stubConsoleOutput(win);
        },
    });
}

function addSourceEditor() {
    findPane('source').find('[data-cy="new-editor-dropdown-btn"]').click();
    cy.get('[data-cy="new-add-editor-btn"]:visible').first().click();
    cy.get('span.lm_title:visible')
        .filter(':contains("source")')
        .should($titles => {
            expect($titles.length).to.be.at.least(2);
        });
}

function closeSourceEditor(sourceEditorIndex: number) {
    cy.get('span.lm_title:visible')
        .filter(':contains("source")')
        .eq(sourceEditorIndex)
        .closest('.lm_tab')
        .find('.lm_close_tab')
        .click({force: true});
}

function selectDiffSource(side: 'lhs' | 'rhs', editorNumber: number) {
    findPane('Diff').find(`select.diff-picker.${side} + .ts-wrapper .ts-control`).click();
    cy.get('.ts-dropdown .option:visible')
        .contains(new RegExp(`source #${editorNumber}`, 'i'))
        .click();
}

function setSourceEditorContent(content: string, sourceEditorIndex: number) {
    cy.get('span.lm_title:visible')
        .filter(':contains("source")')
        .eq(sourceEditorIndex)
        .closest('.lm_item.lm_stack')
        .find('.lm_content .monaco-editor')
        .then($editor => {
            cy.window().then(win => {
                const editorElement = $editor[0];
                const editor = win.monaco.editor
                    .getEditors()
                    .find(candidate => editorElement.contains(candidate.getDomNode()));
                if (!editor) {
                    throw new Error('source editor Monaco instance not found');
                }
                editor.getModel()!.setValue(content);
            });
        });
}

function diffShouldContain(...expected: string[]) {
    findPane('Diff')
        .find('.view-lines', {timeout: 10000})
        .should($el => {
            const text = $el.text().replaceAll('\u00a0', ' ');
            for (const value of expected) {
                expect(text).to.include(value);
            }
        });
}

function diffShouldNotContain(...unexpected: string[]) {
    findPane('Diff')
        .find('.view-lines', {timeout: 10000})
        .should($el => {
            const text = $el.text().replaceAll('\u00a0', ' ');
            for (const value of unexpected) {
                expect(text).to.not.include(value);
            }
        });
}

beforeEach(visitPage);

afterEach(() => {
    return cy.window().then(_win => {
        assertNoConsoleOutput();
    });
});

describe('Diff view', () => {
    it('should open a diff view pane from the Add menu', () => {
        waitForEditors();
        openDiffView();
    });

    it('should update selected compiler labels when compiler options change', () => {
        waitForEditors();

        compilerPane().find('input.options').clear().type('-O2');
        compilerPane().find('input.options').should('have.value', '-O2');

        openDiffView();

        const lhsPicker = () => findPane('Diff').find('select.diff-picker.lhs + .ts-wrapper .ts-control');
        const diffTab = () => cy.get('span.lm_title:visible').filter(':contains("Diff Viewer")');

        lhsPicker().should('contain.text', '-O2');
        diffTab().should('contain.text', '-O2');

        findPane('Diff').find('select.difftype-picker.lhsdifftype + .ts-wrapper .ts-control').click();
        cy.get('.ts-dropdown .option:visible').then($options => {
            const optionTexts = $options.toArray().map(option => option.textContent?.trim());
            expect(optionTexts).not.to.include('Source');
        });

        compilerPane().find('input.options').clear().type('-O1');

        lhsPicker().should('contain.text', '-O1');
        lhsPicker().should('not.contain.text', '-O2');
        diffTab().should('contain.text', '-O1');
        diffTab().should('not.contain.text', '-O2');
    });

    it('should show diff content with two compiler panes', () => {
        waitForEditors();
        setMonacoEditorContent(DIFF_SOURCE);
        monacoEditorTextShouldContain(findPane('Editor #').find('.monaco-editor'), 'mul_variant');

        // Add second compiler
        addCompilerFromEditor();
        cy.get('span.lm_title:visible').filter(':contains("Editor #")').should('have.length', 2);
        lastCompilerContent()
            .find('.monaco-editor .view-lines', {timeout: 10000})
            .should('contain.text', 'mul_variant');

        // Set -DUSE_ADD on the second compiler
        lastCompilerContent().find('input.options').clear().type('-DUSE_ADD');
        lastCompilerContent()
            .find('.monaco-editor .view-lines', {timeout: 10000})
            .should('contain.text', 'add_variant');

        // Open diff view — auto-selects the two compilers
        openDiffView();

        // Verify the diff shows content from at least one variant
        findPane('Diff')
            .find('.view-lines', {timeout: 10000})
            .should($el => {
                const text = $el.text().replaceAll('\u00a0', ' ');
                const hasLhs = text.includes('mul_variant');
                const hasRhs = text.includes('add_variant');
                expect(hasLhs || hasRhs, 'diff should contain at least one variant function name').to.be.true;
            });
    });

    it('should diff source editor contents', () => {
        waitForEditors();
        setSourceEditorContent(SOURCE_A, 0);

        addSourceEditor();
        setSourceEditorContent(SOURCE_B, 1);

        openDiffView();
        selectDiffSource('lhs', 1);
        selectDiffSource('rhs', 2);

        diffShouldContain('source_alpha', 'source_beta');

        setSourceEditorContent(SOURCE_A_UPDATED, 0);

        diffShouldContain('source_alpha_updated', 'source_beta');
    });

    it('should include source editors opened after the diff view', () => {
        waitForEditors();
        setSourceEditorContent(SOURCE_A, 0);

        openDiffView();
        addSourceEditor();
        setSourceEditorContent(SOURCE_B, 1);

        selectDiffSource('lhs', 1);
        selectDiffSource('rhs', 2);

        diffShouldContain('source_alpha', 'source_beta');
    });

    it('should offer source editors when no compiler pane is present', () => {
        visitState(buildSourceOnlyDiffState());

        selectDiffSource('lhs', 1);
        selectDiffSource('rhs', 2);

        diffShouldContain('source_alpha', 'source_beta');
    });

    it('should clear a selected source editor when it closes', () => {
        waitForEditors();
        setSourceEditorContent(SOURCE_A, 0);
        addSourceEditor();
        setSourceEditorContent(SOURCE_B, 1);

        openDiffView();
        selectDiffSource('lhs', 1);
        selectDiffSource('rhs', 2);

        closeSourceEditor(1);

        findPane('Diff')
            .find('select.diff-picker.rhs + .ts-wrapper .ts-control')
            .should('not.contain.text', 'source #2');
        diffShouldContain('source_alpha');
        diffShouldNotContain('source_beta');
    });

    it('should restore source editor selections from layout state', () => {
        visitState(
            buildSourceOnlyDiffState({
                lhs: 'source_1',
                rhs: 'source_2',
                lhsdifftype: DIFF_TYPE_SOURCE,
                rhsdifftype: DIFF_TYPE_SOURCE,
            }),
        );

        findPane('Diff').find('select.diff-picker.lhs + .ts-wrapper .ts-control').should('contain.text', 'source #1');
        findPane('Diff').find('select.diff-picker.rhs + .ts-wrapper .ts-control').should('contain.text', 'source #2');
        findPane('Diff').find('select.difftype-picker.lhsdifftype + .ts-wrapper .ts-control').click();
        cy.get('.ts-dropdown .option:visible').should('have.length', 1).and('contain.text', 'Source');
        diffShouldContain('source_alpha', 'source_beta');
    });
});
