import {it} from 'mocha';


describe('UI testing', () => {
    beforeEach(() => {
        cy.visit('/');
        cy.get('[data-cy="close-alert-btn"]:visible').click();
        cy.get('[data-cy="new-compiler-dropdown-btn"]:visible').click();
        // Shows every pane button even if the compiler does not support it
        cy.get('[data-cy="new-pane-dropdown"]:visible button').each(($btn) => {
            $btn.prop('disabled', false).show();
        });
    });

    afterEach(() => {
        cy.window().then((win) => {
            // Makes sure the pane did not output anything to stderr
            expect(win.console.error).to.have.callCount(0);
            expect(win.console.warn).to.have.callCount(0);
        });
        // Can't bring the site to a default state by clearing localStorage, so do it manually
        cy.get('[data-cy="more-dropdown-btn"]').click();
        cy.get('[data-cy="reset-ui-btn"]:visible').click();
    });

    function addPaneOpenTest(datacy, title) {
        it(title + " pane", () => {
            cy.get(`[data-cy="new-${datacy}-btn"]:visible`).click();
            // Not the most consistent way, but the easiest one!
            cy.get('span.lm_title:visible').contains(title);
        });
    }

    addPaneOpenTest("create-executor", "Executor");
    addPaneOpenTest("view-optimization", "Opt Viewer");
    addPaneOpenTest("view-pp", "Preprocessor");
    addPaneOpenTest("view-ast", "Ast Viewer");
    addPaneOpenTest("view-ir", "LLVM IR");
    addPaneOpenTest("view-llvm-opt-pipeline", "Pipeline");
    addPaneOpenTest("view-device", "Device");
    addPaneOpenTest("view-rustmir", "MIR");
    addPaneOpenTest("view-rusthir", "HIR");
    addPaneOpenTest("view-rustmacroexp", "Macro");
    addPaneOpenTest("view-haskellCore", "Core");
    addPaneOpenTest("view-haskellStg", "STG");
    addPaneOpenTest("view-haskellCmm", "Cmm");
    addPaneOpenTest("view-gccdump", "Dump");
    addPaneOpenTest("view-gnatdebugtree", "Tree");
    addPaneOpenTest("view-gnatdebug", "Debug");
    addPaneOpenTest("view-cfg", "CFG");


    // runFrontendTest('ui');
});
