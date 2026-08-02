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
    findPane,
    monacoEditorTextShouldContain,
    monacoEditorTextShouldNotContain,
    setMonacoEditorContent,
    setupAndWaitForCompilation,
    visitPage,
    waitForCompilationToSettle,
} from '../support/utils';

/**
 * These tests build a C++ -> C chain: the first pane preprocesses C++, and the pane below it
 * compiles that output as C. C is deliberate — it does not mangle, so the assembly it emits
 * carries plain `square:` labels. (Chaining straight into an assembler instead would not work
 * out of the box: CE emits Intel-syntax assembly with demangled `square(int):` labels, which
 * GNU AS rejects. That friction is the feature's own subject matter, not something to assert
 * here.) Deeper links reuse C, which the downstream picks up automatically once the chain
 * output language has been switched once.
 */
const PREPROCESS = '-E';
/** `-P` additionally suppresses the `# 1 "file"` linemarkers, giving a visibly different output. */
const PREPROCESS_NO_LINEMARKERS = '-E -P';

/** The 🔗 marker in a pane title means "chained to another compiler" (Compiler.getPaneTag). */
const CHAIN_MARKER = '🔗';

const CUBE_SOURCE = 'int cube(int num) {\n    return num * num * num;\n}\n';
/** Fails during preprocessing, so it fails even for a pane running `-E`. */
const UNPREPROCESSABLE_SOURCE = '#include <no_such_header_xyzzy.h>\nint square(int num) { return num * num; }\n';

/** Visible tabs of every compiler pane, chained or not, in layout order. */
function compilerTabs() {
    return cy.get('span.lm_title:visible').filter(':contains("Editor #")');
}

/** Visible tabs of only the chained compiler panes. */
function chainedTabs() {
    return cy.get('span.lm_title:visible').filter(`:contains("${CHAIN_MARKER}")`);
}

/** Content area of the nth compiler pane (0-based, layout order). */
function paneAt(index: number) {
    return compilerTabs().eq(index).closest('.lm_item.lm_stack').find('.lm_content');
}

/** Monaco output editor of the nth compiler pane. */
function paneOutputAt(index: number) {
    return paneAt(index).find('.monaco-editor');
}

/**
 * Wait for one specific pane to reach a terminal compile state. `waitForCompilationToSettle`
 * only checks that *some* pane has settled, which is not enough once a chain is in play —
 * and closing a pane with a request still in flight aborts it, which surfaces as an unhandled
 * rejection from the application.
 */
function waitForPaneToSettle(index: number) {
    paneAt(index).find('.status-icon.fa-check-circle, .status-icon.fa-times-circle', {timeout: 30_000}).should('exist');
}

/** Set the compiler options of the nth compiler pane and wait for the recompile. */
function setCompilerOptions(index: number, options: string) {
    paneAt(index).find('input.options').first().should('be.visible').clear().type(`${options}{enter}`);
    waitForCompilationToSettle();
}

/**
 * Declare the language the nth pane's output is in, which is what its downstreams compile as.
 * The picker only exists once that pane has a downstream, and renders its dropdown on <body>.
 * Options are matched exactly so that "C" does not select "C++" or "C with Coccinelle".
 */
function setChainLanguage(index: number, languageName: string) {
    // The picker is created lazily when the upstream learns it has a downstream, so wait for
    // it rather than racing it.
    paneAt(index).find('.chain-language-select .ts-control').should('be.visible').click();
    cy.get('.ts-dropdown .option:visible')
        .filter((_i, el) => el.innerText.trim() === languageName)
        .first()
        .click();
    waitForCompilationToSettle();
}

/** The source editor pane's own language picker control. */
function editorLanguagePicker() {
    return findPane('source').find('select.change-language + .ts-wrapper .ts-control');
}

/** The nth compiler pane's chain output language picker control. */
function chainLanguagePicker(index: number) {
    return paneAt(index).find('select.change-chain-language + .ts-wrapper .ts-control');
}

/**
 * Open a TomSelect language picker and capture the language names it offers under `alias`.
 * The options only exist in the DOM while the dropdown is open, and it renders on <body>.
 */
function captureLanguageOptions($control: JQuery<HTMLElement>, alias: string) {
    cy.wrap($control).click();
    cy.get('.ts-dropdown .option:visible')
        .should('have.length.greaterThan', 0)
        .then($options => $options.toArray().map(option => option.innerText.trim()))
        .as(alias);
}

/** Open a new chained compiler taking its source from the nth compiler pane's output. */
function chainFrom(index: number) {
    paneAt(index).find('[data-cy="new-compiler-dropdown-btn"]').first().should('be.visible').click();
    cy.get('[data-cy="new-add-chained-compiler-btn"]:visible').first().click();
    waitForCompilationToSettle();
}

/** Clone the nth compiler pane via its "Add new" dropdown. */
function cloneFrom(index: number) {
    paneAt(index).find('[data-cy="new-compiler-dropdown-btn"]').first().should('be.visible').click();
    cy.get('[data-cy="new-add-compiler-btn"]:visible').first().click();
    waitForCompilationToSettle();
}

/**
 * Close the nth compiler pane via its tab's close button. Forced because with several panes
 * side by side the tabs are narrow enough that the stack's own close button overlaps the
 * per-tab one; we specifically want to close this tab, not the whole stack.
 */
function closePaneAt(index: number) {
    compilerTabs().eq(index).closest('.lm_tab').find('.lm_close_tab').click({force: true});
}

describe('Chained compiler panes', () => {
    beforeEach(() => {
        visitPage();
        setupAndWaitForCompilation();
        // Pane 0 is the editor-backed C++ compiler from the default layout. Preprocessing its
        // source gives the pane below it something it can compile in a different language.
        setCompilerOptions(0, PREPROCESS);
        chainFrom(0);
        setChainLanguage(0, 'C');
    });

    it('chains a pane to the upstream output and propagates edits down the chain', () => {
        // Ordered from the structural outwards, so that a failure part-way through still
        // implies everything checked before it held.

        // The link itself: A gained a downstream, and only that downstream is marked chained.
        compilerTabs().should('have.length', 2);
        chainedTabs().should('have.length', 1);

        // The chain output language belongs to the pane whose output is being consumed, so
        // only A offers the picker; B has no downstream of its own yet.
        paneAt(0).find('.chain-language-select').should('be.visible');
        paneAt(1).find('.chain-language-select').should('not.be.visible');

        // B compiled what A handed it, in the language A declared: the function reaches B's
        // assembly output under a plain `square:` label, since C does not mangle.
        monacoEditorTextShouldContain(paneOutputAt(1), 'square:');

        // Extend to editor -> A (C++, -E) -> B (C, -E) -> C (C) and edit at the top.
        setCompilerOptions(1, PREPROCESS);
        chainFrom(1);
        compilerTabs().should('have.length', 3);

        setMonacoEditorContent(CUBE_SOURCE);
        waitForCompilationToSettle();

        // The edit has to travel editor -> A -> B -> C to reach the far end.
        monacoEditorTextShouldContain(paneOutputAt(0), 'cube');
        monacoEditorTextShouldContain(paneOutputAt(1), 'cube');
        monacoEditorTextShouldContain(paneOutputAt(2), 'cube');
    });

    it('offers the same languages for chain output as the editor offers for source', () => {
        // Locate both controls before opening either. They are found via their pane's tab
        // title, and an open dropdown seems to shift the layout enough that those lookups stop
        // matching, so everything past this point works off held element references.
        editorLanguagePicker().then($editorControl => {
            chainLanguagePicker(0).then($chainControl => {
                captureLanguageOptions($editorControl, 'editorLanguages');
                captureLanguageOptions($chainControl, 'chainLanguages');

                cy.get('@editorLanguages').then(editorLanguages => {
                    cy.get('@chainLanguages').then(chainLanguages => {
                        // CMake is left out of the comparison on purpose. The server injects it
                        // unconditionally so tree/IDE mode can resolve CMakeLists.txt, even
                        // though it has no compilers of its own, which makes it a candidate for
                        // being dropped from the chain picker later; ignoring it keeps this
                        // test valid either way.
                        const withoutCmake = (names: unknown) =>
                            (names as string[]).filter(name => name !== 'CMake').sort();
                        expect(withoutCmake(chainLanguages), 'chain output languages').to.deep.equal(
                            withoutCmake(editorLanguages),
                        );
                    });
                });
            });
        });
    });

    it('reports upstream failure downstream without issuing a compile request for it', () => {
        cy.intercept('POST', '**/api/compiler/**/compile*').as('compileRequest');

        setMonacoEditorContent(UNPREPROCESSABLE_SOURCE);
        waitForCompilationToSettle();

        monacoEditorTextShouldContain(paneOutputAt(1), '<Upstream compilation failed>');

        // The downstream reports the failure locally, so it must not have sent the upstream's
        // failure text off to be compiled.
        cy.get('@compileRequest.all').then(interceptions => {
            const wasted = (interceptions as unknown as {request: {body?: {source?: string}}}[]).filter(i =>
                (i.request.body?.source ?? '').includes('Upstream compilation failed'),
            );
            expect(wasted, 'no compile request for the upstream failure text').to.have.length(0);
        });
    });

    it('turns the next pane into an editor when an intermediate pane is closed', () => {
        // editor -> A -> B -> C -> D. C strips the linemarkers, which makes its own output
        // distinguishable from the input B hands it — both otherwise contain `square`.
        setCompilerOptions(1, PREPROCESS);
        chainFrom(1);
        setCompilerOptions(2, PREPROCESS_NO_LINEMARKERS);
        chainFrom(2);
        compilerTabs().should('have.length', 4);
        chainedTabs().should('have.length', 3);

        // Close B, the first intermediate pane.
        closePaneAt(1);

        // Two compiler panes go: B is destroyed, and C is replaced in place by an editor.
        compilerTabs().should('have.length', 2);
        // C is now an editor, in the language its upstream declared its output to be...
        cy.contains('span.lm_title:visible', 'C source #2').should('exist');
        // ...and D rebound to that editor rather than being converted in turn, so it survives
        // as an ordinary editor-backed compiler and nothing remains chained.
        chainedTabs().should('have.length', 0);

        // The editor holds what C was displaying — C's own output, with the linemarkers it
        // stripped — and not the input B was feeding it, which still has them.
        monacoEditorTextShouldContain(findPane('C source #2').find('.monaco-editor'), 'square');
        monacoEditorTextShouldNotContain(findPane('C source #2').find('.monaco-editor'), '# 1 "');

        // B was A's only downstream, so A no longer has anything to declare an output language
        // for and must stop offering the picker.
        paneAt(0).find('.chain-language-select').should('not.be.visible');
    });

    it('does not recompile a downstream when the upstream re-sends an identical result', () => {
        // Give B options of its own, so the second downstream added below does not issue the
        // identical request and land on the frontend cache, which would muddy the reading.
        setCompilerOptions(1, PREPROCESS_NO_LINEMARKERS);
        waitForPaneToSettle(1);

        paneAt(1)
            .find('.compile-info')
            .invoke('text')
            .then(before => {
                // It compiled for real, so the label carries a timing rather than "cached".
                expect(before, 'B should have compiled for real first').to.match(/\dms/);

                // Adding a second downstream makes A push its current result to *every*
                // downstream, so B is handed a result it has already consumed.
                chainFrom(0);
                waitForPaneToSettle(2);

                // B's derived source has not changed, so it must not compile again — the label
                // would change (to "- cached", or to a fresh timing) if it had. Every needless
                // recompile in a chain is one the project pays for.
                paneAt(1).find('.compile-info').should('have.text', before);
            });
    });

    it('clones a chained pane, keeping the copy independent and both chains fed', () => {
        // Ordered from the structural outwards, as above.

        // editor -> A (C++, -E) -> B (C, -E) -> C (C), then clone B.
        setCompilerOptions(1, PREPROCESS);
        chainFrom(1);
        compilerTabs().should('have.length', 3);

        cloneFrom(1);

        // The copy keeps B's chain link — it too is fed by A — instead of falling back to
        // being editor-backed, so all three downstream panes are chained. Only B is copied:
        // C is not duplicated.
        compilerTabs().should('have.length', 4);
        chainedTabs().should('have.length', 3);
        // ...and the copy starts out with B's options.
        paneAt(3).find('input.options').first().should('have.value', PREPROCESS);

        // Change the copy so its output differs from B's: dropping the linemarkers is visible
        // in the output text.
        setCompilerOptions(3, PREPROCESS_NO_LINEMARKERS);

        paneAt(3).find('input.options').first().should('have.value', PREPROCESS_NO_LINEMARKERS);
        monacoEditorTextShouldNotContain(paneOutputAt(3), '# 1 "');
        // The change is confined to the copy: B keeps its own options and output, and C, which
        // is fed by B and not by the copy, is unaffected.
        paneAt(1).find('input.options').first().should('have.value', PREPROCESS);
        monacoEditorTextShouldContain(paneOutputAt(1), '# 1 "');
        monacoEditorTextShouldContain(paneOutputAt(2), 'square');

        // A now feeds two separate chains; an edit must still reach the end of both.
        setMonacoEditorContent(CUBE_SOURCE);
        waitForCompilationToSettle();

        monacoEditorTextShouldContain(paneOutputAt(2), 'cube');
        monacoEditorTextShouldContain(paneOutputAt(3), 'cube');
    });
});

/**
 * These layouts cannot be produced through the UI — each chain link always creates a fresh
 * pane. They guard against a hand-edited or hand-crafted permalink forming a cycle, where each
 * pane would feed the other for ever and bill the server for a compilation every round.
 */
describe('Chained compiler cycle guards', () => {
    const SOURCE = 'int square(int num) {\n    return num * num;\n}\n';
    /** Generous: a settling layout legitimately compiles a handful of times. A loop would not stop. */
    const RUNAWAY_THRESHOLD = 12;

    function craftedState(compilers: {id: number; sourceCompiler: number}[]) {
        return {
            version: 4,
            content: [
                {
                    type: 'row',
                    content: [
                        {
                            type: 'component',
                            componentName: 'codeEditor',
                            componentState: {id: 1, lang: 'c++', source: SOURCE},
                            isClosable: true,
                        },
                        ...compilers.map(c => ({
                            type: 'component',
                            componentName: 'compiler',
                            componentState: {
                                id: c.id,
                                compiler: 'gdefault',
                                lang: 'c++',
                                sourceCompiler: c.sourceCompiler,
                            },
                            isClosable: true,
                        })),
                    ],
                },
            ],
        };
    }

    function visitCrafted(compilers: {id: number; sourceCompiler: number}[]) {
        cy.intercept('POST', '**/api/compiler/**/compile*').as('compileRequest');
        cy.visit(`/#${serialiseState(craftedState(compilers))}`);
        cy.get('span.lm_title:visible', {timeout: 30_000}).should('exist');
        // Give any runaway loop room to make itself obvious before counting.
        cy.wait(5000);
    }

    function shouldNotHaveRunAway() {
        cy.get('@compileRequest.all').should(requests => {
            expect(
                (requests as unknown as unknown[]).length,
                'compilations should stop, not loop for ever',
            ).to.be.lessThan(RUNAWAY_THRESHOLD);
        });
    }

    it('drops a compiler chained to itself', () => {
        visitCrafted([{id: 1, sourceCompiler: 1}]);

        // The self-link is discarded at construction, so the pane comes up editor-backed.
        chainedTabs().should('have.length', 0);
        shouldNotHaveRunAway();
    });

    it('severs a two-pane chain cycle', () => {
        visitCrafted([
            {id: 1, sourceCompiler: 2},
            {id: 2, sourceCompiler: 1},
        ]);

        // At least one side must be detached; if both stayed chained they would feed each
        // other for ever.
        chainedTabs().should('have.length.lessThan', 2);
        shouldNotHaveRunAway();
    });
});
