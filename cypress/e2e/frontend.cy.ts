// eslint-disable-next-line node/no-unpublished-import
import {it} from 'mocha';

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
    dump: {name: 'Tree/RTL', selector: 'view-gccdump'},
    tree: {name: 'Tree', selector: 'view-gnatdebugtree'},
    debug: {name: 'Debug', selector: 'view-gnatdebug'},
    cfg: {name: 'CFG', selector: 'view-cfg'},
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

    afterEach('Ensure no output in console', () => {
        return cy.window().then(win => {
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

describe('Known good state test', () => {
    beforeEach(() => {
        cy.visit(
            // This URL manually created. If you need to update, run a local server like the docs/UsingCypress.md say,
            // then paste this into your browser, make the changes, and then re-share as a full link.
            // eslint-disable-next-line max-len
            '/#z:OYLghAFBqd5TB8IAsQGMD2ATApgUWwEsAXTAJwBoiQIAzIgG1wDsBDAW1xAHIBGHpTqYWJAMro2zEHwAsQkSQCqAZ1wAFAB68ADIIBWMyozYtQ6AKQAmAELWblNc3QkiI2q2wBhTIwCuHCwgVpSeADJELLgAcgEARrjkIPIADpgqpG4sPv6BwZRpGa4iEVGxHAlJ8k64LlliJGzkJDkBQSE1dSINTSSlMfGJyY6Nza15HaN9kQMVQ7IAlI6YfuTo3DwA9JsA1AAqAJ4puDsHK%2BQ7WHg7KIm4lDsUO4yYbNg7pju4mpwpzAB0Fh0AEFIiQdioAI5%2BJq4CBgnYsAILHYWADsdhBO2xO3IuBIqxYiICOwAVMSOBYAMyY4HogAiPCWjF4AFZBEEeHpKJheF57PYIed1qirFSBJQSLomUsANYgVk6Yy8WSCDgKpWc7m8niCFQgJVSrlMyhwWBoLAsYTkDimdbUDzEMjkIjYIwmMwASTdlls9mWq3WvG2%2ByOJzOq0uOBOtzxDyeLzeHyJ31%2BAKBoNEEOhsPhWaRHBR6NpONx%2BMJFLJFOptIZJpZPHZlC1gh1PitFFtLBFADUiLgAO6JHYQQikJ7WcULQRGvQLJa3N5DCDMlVq4Ks5vSnm8PUGyXSpZmqBoch%2BFQkFBEKg0CBYDgpJiJaKcDbAAW2HZ4OhsPyMEiCLgToUK6RjCKIEhSNwcgKKIqgaNoxqUIYfCOLgzjFEEECeOMQSoeEMzlJURiFJkIi4SR6RkSw/REUMqGdJhPRjL4bRGIx9RTLRgxJAxUwUXxvTcXMvFLCQeK4KBOiro2HLbjqABK57ggAEp6Ck7H2g7Du%2Bfqft%2Bv7/iOY7OqK4oPD4D5Phck58NOB7GvOlCLngSQrsqPCqpQ6pWJuLY7rqjj7rOMqUPKshKg2VJyUhOozoepqIMeqAYJgVnMDetD3o%2BGUgMABl/iQlB4AAbkQ6xaQOADyxz%2BQw/6JPqEBxNucSRE0By8BKbXsOQBxVXE%2Bi1EaEr3lwohVSwjCdUheBxH4wBeFIjD6vwgh4F2wDSLN17DUQJW4Kt3LfLUfgkBsEpguh26MEQcTkB1Ph4Nu4lEOqa1LHQJjAColU1awXWCOB4iSNIMHA/BWjbihximOYH4OLdcT6pASyYCkmGra2B3kC6eAo%2B5HHuNhLDeKxeT4aTwnEahpGYQJBRUZh1P0WhGGcb0DNEywzHTGUPHsfx5N4SMQmEQLdkBms0GUAOpgkDVJCeh2CUNk2/k6jsul2PpuA/oVxnATZYp2fFjkLrgS5uXKGoeV5Pl%2BfJu5BYaCXJWgF5sOgsrnmwwD3Le2XWS%2BXC8NrgoFf%2BgFG6BqEwyowobCGhzHKc5xRtcsb3I8FyJu8nypg%2B6YggiUIwniebggWRYYhmpZ4gS5BEgWVYFjWGZ1pKElSTJ6tOzwPTezsqh%2ByclU6QjX564Z4KjkbZmoTslk5cOtn2SFTkucuMn2xuW6xc7%2Bqu45iXIGgKwkCkZ0OneaUr%2BQIdvpPkcAaEMduqhwOQWD8gQ2oUNIRhgOB6KRAa9xitqXgVUzpX3BJgOgWtn7TwNhAZe1kF7r0PBbK2tAd7rl8vvSBgUj4OTnKfc0IAzwXltOgcgmBvgpBvkHDKj8w5IP1lHN%2B448ZgUUN/aCv9FCQ0QtyeO6E9rExwsLIwBF%2BYiUokULIDM6ZZBZrxNmEieZC1yCLbmvM1GC05tIwSzQDGS3ErgSSbppIeT7gfHgSkLw7AALJezoTsfAmgUimAyCITS/YhwXHDrrDhs8TIThNhZO%2B6C15mznNg1yuC7b4MdvYvcx94lhRAAANkirwaKhDWzOw3jbKkaJ/hijFGiVkABONENS%2BB8CsFYNEHkrDrj4DoTU/c4kmmSklAZZ8QAnXQGdCgN8mg/XUKYdCYgUCYAHJyUa0STCYWmVERgcyFn%2BWYUMfKyD/yUF2Q/V8ztjnQPEPMxZ24RnAnID9Z2IyGjEE5EDPhoMBGwWUP/ERBh3RwwwAjYwd0CZowxlkLGPIcZ40OvAJYCdAwywvC89ZsyrlLJnBJQGssQFgNsRAopPBPGnVMsEmwU9Qkjn5HpBwHj56xNIaFLe1ssmKmSd5PeGtD7BTdklNAwB2AkDmgtCxAcsorOfKcngZKKUz2jtw2OXz%2BEyEEXBH50N2LiK6FhKROiZFU3FvI2mTMlHGMZookohqaYaO1bzLmWqmJcStazT2LE9UmL5rMGmYlu7WPAYUgKABxaIwI9g7HpLgeawB9gSX8dpIJ7CZ6G24QvKJ6VV4m0webZyltEnuQbLvAhXLiE8pPu7EA14mESpOaHaVibCryudIq%2BOidgy7BTuGdOVwYx3HjLnV4%2BcUw/CLrgQEJcsxl1zAiauqJa5YhxA3CsLdyRtxpB3NEjIu6WNAh9fFAadRhDCD2ZxOx1JxsCYgmlsqUHhONuZJe1aMG9M3rm7e7KHb7u5RkvpfKQAoDYCoWU6FGA%2BDjIHatrC61Xpfo2kCH8lUfJVV84RGqGIOqyB4UmDNZFetZio8iZr8M0WdeovR2i2JofZt0J1cjrWupaGa%2BjZifXbr9Xu4tgaVJeAfXic9E9oMHLCfSyJD7013tNoyl9OD81rg5UWnpLtJM2zZVFAlAVn3kKQBQ9GRVwNicgzKmDXCm3wZbYittoZU4RguN2m4vac7PAHcmL4w6/ijozKXHMFdp3IlnSWBd5Ym6VhXQEduIJO6ip3QIf1xbFZ8YTQJylc8U22TTffJ9Smc3SeU3kzyKTP0lu/UeX9Aq2BCsjQtKt%2BmpWGcE7Bnhn93lQSQ3/BCqGbWYUw2Td1oQDW0bwyagjPWiNmI6xzN1FGxvUbFv10j5GKai1MSRmQLGrEGhi/3YNobw0VejePBLOtyUv2TaZVLon0sMpKVlvNeC5OpKIekyTmmUqMEYCVDgit1BEGOLdKIlUb48AALR0BYJgQHxAVCSHINgQHJUpB%2BFwID9gXAVA8G3Mcgz9bOEbThswQHKgDgVF8EddaAQUjA7/IwQHHAcB/guvVxV9VzrkEB5ES0ahoSsFcFIQH3iVBqFWhKL%2BiHwZCPVUhNAr33ufe%2B%2BhGY/2xEYSAgAMT8N2TrxCHVAWVngbQSoMhzSaAAdVdJeDchpfV4rVmpg9R6T1xa%2Bz9mY8XL2HevUZZLp2RNoIyhlq7zKkkFvy8Wx7G9ntoH/YB4DDRgBVfvpjxLcrjNwd4RBEXqrvltcAZqqjOqsNmpw3RdRRHlGDeI7NnPmi7WMfQ9NpbFePVcxo7h0SW61s2OtwVzj3GxB7EDS72rSXb2pvOzEzNz6ElvqD3dgroesGstywUkPmX5SslkP8Pg2St/b539v%2BQDZ2kcs6d0tJT3%2BlDJSIwwOHpgDA87GV7FQEFXwaZ4kQHzKScIea6LtVWfRFGEizdF3U72LXUDxBSDoXWH5yeAuVgVdwjkExOwiXvR9wzSnAn2uzfXlCsBqQ3133wK3wAA531OUFMSEw9y10A6BY89N48assdX4n8TMjBFdhoVc1dtVeBJ1vNRAFheA28e52N%2B4vBld%2B9B8k1PdkDF5UDxMs1MkA8ZM8sZ9l9yDeUhlSpyoxUIANCgxAomCU9Gs09v8M8UNs9KNNEutsM%2BsW8FFqJS8LVy8bDzDbV5tdFa8tEZsnDFsGMesmNltzFLdgC2QbdeAI0ypewAl%2BM3djtJDxM0sx90DMsFDbsP0VDS0yFy1I8gNXsvAOAOA49g56DE8G1k8Gsv8f5kNxd/9nDOsSZutJtC8JZzU7DCMy9RsyMjFfD3D9F/DvCm9PCi8VsBC2MQDNsuMl48iB8GCkC4jR9fdLt59kiSD5NT9VDs1sDCCN8rBGlCDZAykdi9i0QqQPIl8FMKDBkKEANdNxVqta1xCSj9CyizNpYLMO005IxbMs4%2B1HMkwC5XNi5MxwRuC4QfNCw/M64AtG5m4SQQtKQ11wsN1MVWMrdgiCtgQnF9t4CQkJDh8zsZC/dFjX0WVp91Q2U0iitw9KFlIOBK1aDCi7iGCGcX8msKjWsAFqips896iFtGijVmj6ZWiHD2jujXDK8XCBimj6N%2Bj68nDVtBDRj7FHFwRnEz1MT7iPdcTvdH0Fjs0liSTghsl/gdAxQqQGlCCmlCDsk18aliDySz8St0B0BsAycJlsBsA8R%2BdFYMM1pKBlovT3AfS4gXhvYVB/SMUjk0ouxsAAB9CAvWIgbQdyDHIo6IurYqfENgJgUM8FAMiUR4xnJ8ZHXAaMlQPwOgBgRMz6Fkz5Nk35agbxekMndrX7VgTAMM7FIspJB6aqHM8M5oRgRsh8bcRiICaZfnJJbmSwgvawwY41Bw%2Bw6iYU3PavLo5c5vWcvoxjdciWeFW6D7Xs7FT2EgbMjXCUUVQc0BJCQA/cPwV0dsn0kqdGE8706LIQ%2BxQNLwbjPYCSTYBSPYMIKY4ojU4TFA7U8fJIokwPWTHyKkf4OpLfHQc0wgtEPgKkWQRUEIO0q7eUPgcpFC7Y2QQgmpbJAioikik4kIwKbCnJKwf4NEbJeiqwCKKwRC5CtEffXgQ/dUY/WfTLA/SijAnGXxIIWQIAA%3D%3D',
            {
                onBeforeLoad: win => {
                    stubConsoleOutput(win);
                },
            },
        );
    });

    afterEach('Ensure no output in console', () => {
        return cy.window().then(win => {
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
