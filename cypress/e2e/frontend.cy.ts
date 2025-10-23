import {serialiseState} from '../../shared/url-serialization.js';
import {assertNoConsoleOutput, stubConsoleOutput} from '../support/utils';

const PANE_DATA_MAP = {
    executor: {name: 'Executor', selector: 'create-executor'},
    opt: {name: 'Opt Viewer', selector: 'view-optimization'},
    stackusage: {name: 'Stack Usage Viewer', selector: 'view-stack-usage'},
    preprocessor: {name: 'Preprocessor', selector: 'view-pp'},
    ast: {name: 'Ast Viewer', selector: 'view-ast'},
    llvmir: {name: 'LLVM IR', selector: 'view-ir'},
    pipeline: {name: 'Pipeline', selector: 'view-opt-pipeline'},
    device: {name: 'Device', selector: 'view-device'},
    mir: {name: 'MIR', selector: 'view-rustmir'},
    hir: {name: 'HIR', selector: 'view-rusthir'},
    macro: {name: 'Macro', selector: 'view-rustmacroexp'},
    core: {name: 'Core', selector: 'view-haskellCore'},
    stg: {name: 'STG', selector: 'view-haskellStg'},
    cmm: {name: 'Cmm', selector: 'view-haskellCmm'},
    clojure_macro: {name: 'Clojure Macro', selector: 'view-clojuremacroexp'},
    dump: {name: 'Tree/RTL', selector: 'view-gccdump'},
    tree: {name: 'Tree', selector: 'view-gnatdebugtree'},
    debug: {name: 'Debug', selector: 'view-gnatdebug'},
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
    addPaneOpenTest(PANE_DATA_MAP.preprocessor);
    addPaneOpenTest(PANE_DATA_MAP.ast);
    addPaneOpenTest(PANE_DATA_MAP.llvmir);
    addPaneOpenTest(PANE_DATA_MAP.pipeline);
    // TODO: re-enable this when fixed addPaneOpenTest(PANE_DATA_MAP.device);
    addPaneOpenTest(PANE_DATA_MAP.mir);
    addPaneOpenTest(PANE_DATA_MAP.hir);
    addPaneOpenTest(PANE_DATA_MAP.macro);
    addPaneOpenTest(PANE_DATA_MAP.core);
    addPaneOpenTest(PANE_DATA_MAP.stg);
    addPaneOpenTest(PANE_DATA_MAP.cmm);
    addPaneOpenTest(PANE_DATA_MAP.dump);
    addPaneOpenTest(PANE_DATA_MAP.tree);
    addPaneOpenTest(PANE_DATA_MAP.debug);
    addPaneOpenTest(PANE_DATA_MAP.stackusage);
    addPaneOpenTest(PANE_DATA_MAP.explain);
    // TODO: Bring back once #3899 lands
    // addPaneOpenTest(PaneDataMap.cfg);

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
        // TODO: re-enable this when fixed cy.get('span.lm_title:visible').contains('Conformance');
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

    return {
        version: 4,
        content: [
            {
                type: 'row',
                content: [
                    // Left side: Editor and various views
                    {
                        type: 'column',
                        content: [
                            {
                                type: 'row',
                                width: 37.23237597911227,
                                height: 100,
                                isClosable: true,
                                reorderEnabled: true,
                                content: [
                                    // Editor and conformance/HIR column
                                    {
                                        type: 'column',
                                        width: 40,
                                        isClosable: true,
                                        reorderEnabled: true,
                                        content: [
                                            {
                                                type: 'stack',
                                                width: 50,
                                                height: 50,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'codeEditor',
                                                        componentState: {
                                                            filename: false,
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: editorId,
                                                            lang,
                                                            selection: {
                                                                endColumn: 2,
                                                                endLineNumber: 4,
                                                                positionColumn: 2,
                                                                positionLineNumber: 4,
                                                                selectionStartColumn: 2,
                                                                selectionStartLineNumber: 4,
                                                                startColumn: 2,
                                                                startLineNumber: 4,
                                                            },
                                                            source,
                                                        },
                                                        isClosable: true,
                                                        title: 'C++ source #1',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                            {
                                                type: 'stack',
                                                header: {},
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'conformance',
                                                        componentState: {
                                                            editorid: editorId,
                                                            langId: lang,
                                                            source,
                                                        },
                                                        isClosable: true,
                                                        title: 'Conformance Viewer (Editor #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                            {
                                                type: 'stack',
                                                header: {},
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'rusthir',
                                                        componentState: {
                                                            compilerName: 'g++ default',
                                                            editorid: editorId,
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: compilerId,
                                                            selection: {
                                                                endColumn: 1,
                                                                endLineNumber: 1,
                                                                positionColumn: 1,
                                                                positionLineNumber: 1,
                                                                selectionStartColumn: 1,
                                                                selectionStartLineNumber: 1,
                                                                startColumn: 1,
                                                                startLineNumber: 1,
                                                            },
                                                            treeid: 0,
                                                        },
                                                        isClosable: true,
                                                        title: 'Rust HIR Viewer g++ default (Editor #1, Compiler #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                    // Compiler, stack usage, output, macro expansion column
                                    {
                                        type: 'column',
                                        width: 60,
                                        isClosable: true,
                                        reorderEnabled: true,
                                        content: [
                                            {
                                                type: 'stack',
                                                header: {},
                                                width: 50,
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'compiler',
                                                        componentState: {
                                                            compiler: 'gdefault',
                                                            filters: {
                                                                labels: true,
                                                                binary: false,
                                                                binaryObject: false,
                                                                commentOnly: true,
                                                                debugCalls: false,
                                                                demangle: true,
                                                                directives: true,
                                                                execute: false,
                                                                intel: true,
                                                                libraryCode: true,
                                                                trim: false,
                                                                verboseDemangling: true,
                                                            },
                                                            flagsViewOpen: false,
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: compilerId,
                                                            lang,
                                                            libs: [],
                                                            options: '',
                                                            overrides: [],
                                                            selection: {
                                                                endColumn: 1,
                                                                endLineNumber: 1,
                                                                positionColumn: 1,
                                                                positionLineNumber: 1,
                                                                selectionStartColumn: 1,
                                                                selectionStartLineNumber: 1,
                                                                startColumn: 1,
                                                                startLineNumber: 1,
                                                            },
                                                            source: editorId,
                                                            wantOptInfo: true,
                                                        },
                                                        isClosable: true,
                                                        title: ' g++ default (Editor #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                            {
                                                type: 'stack',
                                                header: {},
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'stackusage',
                                                        componentState: {
                                                            compilerName: 'g++ default',
                                                            editorid: editorId,
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: compilerId,
                                                            selection: {
                                                                endColumn: 1,
                                                                endLineNumber: 1,
                                                                positionColumn: 1,
                                                                positionLineNumber: 1,
                                                                selectionStartColumn: 1,
                                                                selectionStartLineNumber: 1,
                                                                startColumn: 1,
                                                                startLineNumber: 1,
                                                            },
                                                            treeid: 0,
                                                        },
                                                        isClosable: true,
                                                        title: 'Stack Usage Viewer g++ default (Editor #1, Compiler #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                            {
                                                type: 'stack',
                                                header: {},
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'output',
                                                        componentState: {
                                                            compilerName: 'g++ default',
                                                            editorid: editorId,
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: compilerId,
                                                            wrap: false,
                                                        },
                                                        isClosable: true,
                                                        title: 'Output of g++ default (Compiler #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                            {
                                                type: 'stack',
                                                header: {},
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'rustmacroexp',
                                                        componentState: {
                                                            compilerName: 'g++ default',
                                                            editorid: editorId,
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: compilerId,
                                                            selection: {
                                                                endColumn: 1,
                                                                endLineNumber: 1,
                                                                positionColumn: 1,
                                                                positionLineNumber: 1,
                                                                selectionStartColumn: 1,
                                                                selectionStartLineNumber: 1,
                                                                startColumn: 1,
                                                                startLineNumber: 1,
                                                            },
                                                            treeid: 0,
                                                        },
                                                        isClosable: true,
                                                        title: 'Rust Macro Expansion Viewer g++ default (Editor #1, Compiler #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            // Bottom row: More panes Continue with rest of the panes...
                            {
                                type: 'row',
                                content: [
                                    // Left column bottom: executor, gnat, ir, haskell core
                                    {
                                        type: 'column',
                                        width: 50,
                                        isClosable: true,
                                        reorderEnabled: true,
                                        content: [
                                            {
                                                type: 'stack',
                                                header: {},
                                                width: 50,
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'executor',
                                                        componentState: {
                                                            argsPanelShown: false,
                                                            compilationPanelShown: true,
                                                            compiler: 'gdefault',
                                                            compilerName: '',
                                                            compilerOutShown: true,
                                                            execArgs: '',
                                                            execStdin: '',
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: compilerId,
                                                            lang,
                                                            libs: [],
                                                            options: '',
                                                            overrides: [],
                                                            runtimeTools: [],
                                                            source: editorId,
                                                            stdinPanelShown: false,
                                                            tree: false,
                                                            wrap: false,
                                                        },
                                                        isClosable: true,
                                                        title: 'Executor g++ default (C++, Editor #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                            {
                                                type: 'stack',
                                                header: {},
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'gnatdebugtree',
                                                        componentState: {
                                                            compilerName: 'g++ default',
                                                            editorid: editorId,
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: compilerId,
                                                            selection: {
                                                                endColumn: 1,
                                                                endLineNumber: 1,
                                                                positionColumn: 1,
                                                                positionLineNumber: 1,
                                                                selectionStartColumn: 1,
                                                                selectionStartLineNumber: 1,
                                                                startColumn: 1,
                                                                startLineNumber: 1,
                                                            },
                                                            treeid: 0,
                                                        },
                                                        isClosable: true,
                                                        title: 'GNAT Debug Tree Viewer g++ default (Editor #1, Compiler #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                            {
                                                type: 'stack',
                                                header: {},
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'ir',
                                                        componentState: {
                                                            compilerName: 'g++ default',
                                                            editorid: editorId,
                                                            id: compilerId,
                                                            source,
                                                            treeid: false,
                                                        },
                                                        isClosable: true,
                                                        title: 'LLVM IR Viewer g++ default (Editor #1, Compiler #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                            {
                                                type: 'stack',
                                                header: {},
                                                height: 25,
                                                isClosable: true,
                                                activeItemIndex: 0,
                                                reorderEnabled: true,
                                                content: [
                                                    {
                                                        type: 'component',
                                                        componentName: 'haskellCore',
                                                        componentState: {
                                                            compilerName: 'g++ default',
                                                            editorid: editorId,
                                                            fontScale: 14,
                                                            fontUsePx: true,
                                                            id: compilerId,
                                                            selection: {
                                                                endColumn: 1,
                                                                endLineNumber: 1,
                                                                positionColumn: 1,
                                                                positionLineNumber: 1,
                                                                selectionStartColumn: 1,
                                                                selectionStartLineNumber: 1,
                                                                startColumn: 1,
                                                                startLineNumber: 1,
                                                            },
                                                            treeid: 0,
                                                        },
                                                        isClosable: true,
                                                        title: 'GHC Core Viewer g++ default (Editor #1, Compiler #1)',
                                                        reorderEnabled: true,
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                    // Right column bottom: opt, gnat debug, rest of views
                                    {
                                        type: 'column',
                                        width: 50,
                                        isClosable: true,
                                        reorderEnabled: true,
                                        content: [
                                            // Top section: opt, gnatdebug, cfg, explain
                                            {
                                                type: 'row',
                                                content: [
                                                    {
                                                        type: 'column',
                                                        width: 50,
                                                        isClosable: true,
                                                        reorderEnabled: true,
                                                        content: [
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                width: 50,
                                                                height: 25,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'opt',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            'filter-analysis': false,
                                                                            'filter-missed': true,
                                                                            'filter-passed': false,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: false,
                                                                            wrap: false,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'Opt Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 25,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'gnatdebug',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: 0,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'GNAT Debug Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                    {
                                                        type: 'column',
                                                        width: 50,
                                                        isClosable: true,
                                                        reorderEnabled: true,
                                                        content: [
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 25,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'cfg',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: false,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'CFG Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 25,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'explain',
                                                                        componentState: {id: compilerId},
                                                                        isClosable: true,
                                                                        title: 'Claude Explain (Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                ],
                                            },
                                            // Bottom section: device, pp, more views
                                            {
                                                type: 'row',
                                                content: [
                                                    {
                                                        type: 'column',
                                                        width: 33.33,
                                                        isClosable: true,
                                                        reorderEnabled: true,
                                                        content: [
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.33,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'device',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: 0,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'Device Code Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.33,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'pp',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: 0,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'Preprocessor Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.34,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'ast',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: 0,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'Ast Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                    {
                                                        type: 'column',
                                                        width: 33.33,
                                                        isClosable: true,
                                                        reorderEnabled: true,
                                                        content: [
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.33,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'gccdump',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            gccDumpOptions: {},
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: false,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'Tree/RTL Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.33,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'rustmir',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: 0,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'Rust MIR Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.34,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'haskellStg',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: 0,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'STG Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                    {
                                                        type: 'column',
                                                        width: 33.34,
                                                        isClosable: true,
                                                        reorderEnabled: true,
                                                        content: [
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.33,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'haskellCmm',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: 0,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'Cmm Viewer g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.33,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'llvmOptPipelineView',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: false,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'LLVM Opt Pipeline g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                            {
                                                                type: 'stack',
                                                                header: {},
                                                                height: 33.34,
                                                                isClosable: true,
                                                                activeItemIndex: 0,
                                                                reorderEnabled: true,
                                                                content: [
                                                                    {
                                                                        type: 'component',
                                                                        componentName: 'llvmOptPipelineView',
                                                                        componentState: {
                                                                            compilerName: 'g++ default',
                                                                            editorid: editorId,
                                                                            fontScale: 14,
                                                                            fontUsePx: true,
                                                                            id: compilerId,
                                                                            selection: {
                                                                                endColumn: 1,
                                                                                endLineNumber: 1,
                                                                                positionColumn: 1,
                                                                                positionLineNumber: 1,
                                                                                selectionStartColumn: 1,
                                                                                selectionStartLineNumber: 1,
                                                                                startColumn: 1,
                                                                                startLineNumber: 1,
                                                                            },
                                                                            treeid: false,
                                                                        },
                                                                        isClosable: true,
                                                                        title: 'LLVM Opt Pipeline g++ default (Editor #1, Compiler #1)',
                                                                        reorderEnabled: true,
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    };
}

describe('Known good state test', () => {
    beforeEach(() => {
        const state = buildKnownGoodState();
        const hash = serialiseState(state);
        cy.visit(`http://localhost:10240/#${hash}`, {
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

        cy.get('span.lm_title:visible').contains('Output');
        cy.get('span.lm_title:visible').contains('Conformance');
    });
});
