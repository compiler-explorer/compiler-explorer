import {serialiseState} from '../../shared/url-serialization.js';
import {assertNoConsoleOutput, stubConsoleOutput} from '../support/utils';

const PANE_DATA_MAP = {
    codeEditor: {name: 'Editor', selector: 'new-editor'},
    compiler: {name: 'Compiler', selector: 'new-compiler'},
    conformance: {name: 'Conformance', selector: 'new-conformance'},
    output: {name: 'Output', selector: 'new-output-pane'},
    executor: {name: 'Executor', selector: 'create-executor'},
    opt: {name: 'Opt Viewer', selector: 'view-optimization'},
    stackusage: {name: 'Stack Usage Viewer', selector: 'view-stack-usage'},
    pp: {name: 'Preprocessor', selector: 'view-pp'},
    ast: {name: 'Ast Viewer', selector: 'view-ast'},
    ir: {name: 'LLVM IR', selector: 'view-ir'},
    llvmOptPipelineView: {name: 'Pipeline', selector: 'view-opt-pipeline'},
    device: {name: 'Device', selector: 'view-device'},
    rustmir: {name: 'MIR', selector: 'view-rustmir'},
    rusthir: {name: 'HIR', selector: 'view-rusthir'},
    rustmacroexp: {name: 'Macro', selector: 'view-rustmacroexp'},
    haskellCore: {name: 'Core', selector: 'view-haskellCore'},
    haskellStg: {name: 'STG', selector: 'view-haskellStg'},
    haskellCmm: {name: 'Cmm', selector: 'view-haskellCmm'},
    yul: {name: 'Yul', selector: 'view-yul'},
    clojuremacroexp: {name: 'Clojure Macro', selector: 'view-clojuremacroexp'},
    gccdump: {name: 'Tree/RTL', selector: 'view-gccdump'},
    gnatdebugtree: {name: 'Tree', selector: 'view-gnatdebugtree'},
    gnatdebug: {name: 'Debug', selector: 'view-gnatdebug'},
    cfg: {name: 'CFG', selector: 'view-cfg'},
    explain: {name: 'Claude Explain', selector: 'view-explain'},
};

describe('Individual pane testing', () => {
    beforeEach(() => {
        cy.visit('/', {
            onBeforeLoad: win => {
                stubConsoleOutput(win);
            },
        });

        cy.get('[data-cy="new-compiler-dropdown-btn"]:visible').click();
        // Shows every pane button even if the compiler does not support it
        cy.get('[data-cy="new-compiler-pane-dropdown"]:visible button').each($btn => {
            $btn.prop('disabled', false).show();
        });
    });

    afterEach(() => {
        // Ensure no output in console
        return cy.window().then(_win => {
            assertNoConsoleOutput();
        });
    });

    function addPaneOpenTest(paneData) {
        it(paneData.name + ' pane', () => {
            cy.get(`[data-cy="new-${paneData.selector}-btn"]:visible`).click();
            // Not the most consistent way, but the easiest one!
            cy.get('span.lm_title:visible').contains(paneData.name);
        });
    }

    addPaneOpenTest(PANE_DATA_MAP.executor);
    addPaneOpenTest(PANE_DATA_MAP.opt);
    addPaneOpenTest(PANE_DATA_MAP.pp);
    addPaneOpenTest(PANE_DATA_MAP.ast);
    addPaneOpenTest(PANE_DATA_MAP.ir);
    addPaneOpenTest(PANE_DATA_MAP.llvmOptPipelineView);
    addPaneOpenTest(PANE_DATA_MAP.device);
    addPaneOpenTest(PANE_DATA_MAP.rustmir);
    addPaneOpenTest(PANE_DATA_MAP.rusthir);
    addPaneOpenTest(PANE_DATA_MAP.rustmacroexp);
    addPaneOpenTest(PANE_DATA_MAP.haskellCore);
    addPaneOpenTest(PANE_DATA_MAP.haskellStg);
    addPaneOpenTest(PANE_DATA_MAP.haskellCmm);
    addPaneOpenTest(PANE_DATA_MAP.yul);
    addPaneOpenTest(PANE_DATA_MAP.clojuremacroexp);
    addPaneOpenTest(PANE_DATA_MAP.gccdump);
    addPaneOpenTest(PANE_DATA_MAP.gnatdebugtree);
    addPaneOpenTest(PANE_DATA_MAP.gnatdebug);
    addPaneOpenTest(PANE_DATA_MAP.stackusage);
    addPaneOpenTest(PANE_DATA_MAP.explain);
    addPaneOpenTest(PANE_DATA_MAP.cfg);

    it('Output pane', () => {
        // Hide the dropdown
        cy.get('[data-cy="new-compiler-dropdown-btn"]:visible').click();
        cy.get(`[data-cy="new-output-pane-btn"]:visible`).click();
        cy.get('span.lm_title:visible').contains('Output');
    });

    it('Conformance view pane', () => {
        // First, hide the dropdown
        cy.get('[data-cy="new-compiler-dropdown-btn"]:visible').click();
        cy.get('[data-cy="new-editor-dropdown-btn"]:visible').click();
        cy.get('[data-cy="new-conformance-btn"]:visible').click();
        cy.get('span.lm_title:visible').contains('Conformance');
    });
});

// Programmatically built state containing all pane types for comprehensive testing.
// This replaced the hard-coded base64 URL that was 4812 characters long.
// The state is human-readable and maintainable, then serialized to the same compressed format.
function buildKnownGoodState() {
    const editorId = 1;
    const compilerId = 1;
    const lang = 'c++';
    const source = '// Type your code here, or load an example.\nint square(int num) {\n    return num * num;\n}';

    // Helper functions to reduce boilerplate
    const pane = (componentName: string, componentState: any) => ({
        type: 'component',
        componentName,
        componentState,
        isClosable: true,
    });

    const stack = (content: any) => ({
        type: 'stack',
        isClosable: true,
        activeItemIndex: 0,
        content: [content],
    });

    // Define minimal component states for each pane type
    const paneStates: Record<string, any> = {
        codeEditor: {id: editorId, lang, source},
        compiler: {compiler: 'gdefault', id: compilerId, lang, source: editorId},
        conformance: {editorid: editorId, langId: lang, source},
        output: {compiler: compilerId},
        executor: {compiler: compilerId},
        opt: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        stackusage: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        pp: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        ast: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        ir: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        llvmOptPipelineView: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        device: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        rustmir: {compilerName: 'g++ default', editorid: editorId, id: compilerId, treeid: 0},
        rusthir: {compilerName: 'g++ default', editorid: editorId, id: compilerId, treeid: 0},
        rustmacroexp: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        haskellCore: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        haskellStg: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        haskellCmm: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        yul: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        clojuremacroexp: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        gccdump: {
            compilerName: 'g++ default',
            editorid: editorId,
            id: compilerId,
            treeid: 0,
            gccDumpOptions: {},
        },
        gnatdebugtree: {compilerName: 'g++ default', editorid: editorId, id: compilerId, treeid: 0},
        gnatdebug: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
        cfg: {editorid: editorId, id: compilerId},
        explain: {compilerName: 'g++ default', editorid: editorId, id: compilerId},
    };

    // Build all panes from PANE_DATA_MAP
    const allPanes = Object.keys(PANE_DATA_MAP).map(key => stack(pane(key, paneStates[key])));

    // Chunk panes into rows of 8 for a more reasonable layout
    const panesPerRow = 8;
    const rows: {type: string; content: ReturnType<typeof stack>[]}[] = [];
    for (let i = 0; i < allPanes.length; i += panesPerRow) {
        rows.push({
            type: 'row',
            content: allPanes.slice(i, i + panesPerRow),
        });
    }

    return {
        version: 4,
        content: [{type: 'column', content: rows}],
    };
}

describe('Known good state test', () => {
    beforeEach(() => {
        const state = buildKnownGoodState();
        const hash = serialiseState(state);
        cy.visit(`/#${hash}`, {
            onBeforeLoad: win => {
                stubConsoleOutput(win);
            },
        });
    });

    afterEach(() => {
        return cy.window().then(_win => {
            assertNoConsoleOutput();
        });
    });

    it('Correctly loads the page for a state with every pane active', () => {
        for (const paneId in PANE_DATA_MAP) {
            const pane = PANE_DATA_MAP[paneId];
            cy.get('span.lm_title:visible').contains(pane.name);
        }
    });
});
