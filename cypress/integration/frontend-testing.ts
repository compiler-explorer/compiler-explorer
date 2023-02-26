import {it} from 'mocha';
import {assertNoConsoleOutput, stubConsoleOutput} from '../support/utils';

const PANE_DATA_MAP = {
    executor: {name: 'Executor', selector: 'create-executor'},
    opt: {name: 'Opt Viewer', selector: 'view-optimization'},
    preprocessor: {name: 'Preprocessor', selector: 'view-pp'},
    ast: {name: 'Ast Viewer', selector: 'view-ast'},
    llvmir: {name: 'LLVM IR', selector: 'view-ir'},
    pipeline: {name: 'Pipeline', selector: 'view-llvm-opt-pipeline'},
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
                win.localStorage.clear();
            },
        });
        cy.get('[data-cy="new-compiler-dropdown-btn"]:visible').click();
        // Shows every pane button even if the compiler does not support it
        cy.get('[data-cy="new-compiler-pane-dropdown"]:visible button').each($btn => {
            $btn.prop('disabled', false).show();
        });
    });

    afterEach('Ensure no output in console', () => {
        cy.window().then(win => {
            assertNoConsoleOutput();
            win.localStorage.clear();
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
            '/#z:OYLghAFBqd5TB8IAsQGMD2ATApgUWwEsAXTAJwBoiQIAzIgG1wDsBDAW1xAHIBGHpTqYWJAMro2zEHwAsQkSQCqAZ1wAFAB68ADIIBWMyozYtQ6AKQAmAELWblNc3QkiI2q2wBhTIwCuHCwgVpSeADJELLgAcgEARrjkIPIADpgqpG4sPv6BwZRpGa4iEVGxHAlJ8k64LlliJGzkJDkBQSE1dSINTSSlMfGJyY6Nza15HaN9kQMVQ7IAlI6YfuTo3DwA9JsA1AAqAJ4puDsHK%2BQ7WHg7KIm4lDsUO4yYbNg7pju4mpwpzAB0Fh0AEFIiQdioAI5%2BJq4CBgnYsAILHYWADsdhBO2xO3IuBIqxYiICOwAVMSOBYAMyY4HogAiPCWjF4AFZBEEeHpKJheF57PYIed1qirFSBJQSLomUsANYgVk6Yy8WSCDgKpWc7m8niCFQgJVSrlMyhwWBoLAsYTkDimdbUDzEMjkIjYIwmMwASTdlls9mWq3WvG2%2ByOJzOq0uOBOtzxDyeLzeHyJ31%2BAKBoNEEOhsPhWaRHBR6NpONx%2BMJFLJFOptIZJpZPHZlC1gh1PitFFtLBFADUiLgAO6JHYQQikJ7WcULQRGvQLJa3N5DCDMlVq4Ks5vSnm8PUGyXSpZmqBoch%2BFQkFBEKg0CBYDgpJiJaKcDbAAW2HZ4OhsPyMEiCLgToUK6RiGHwAZrBsIaHMcpznFG1yxvcjwXIm7yfKmD7piCCJQjCeJ5uCBZFhiGalniBLkESBZVgWNYZnWkp4rgoH8PWbIctuOoAErnuCAASno8TsfaDsO75%2Bp%2B36/v%2BI5js6orig8PgPk%2BFyTnw04Hsa86UIueBJCuyo8KqlDqlYm4tjuuqOPus4ypQ8qyEqDZUlxxo2TOh6mogx6oBgmBqcwN60Pej4hSAwAyX%2BJBCEwJCJPqEBxNucSRE0By8BK95cKIADyLCMFlnl4F2wDSKV161K4ABuuD6p53y1H4iXZYIYK4A23KMEQcTkJlPh4NuJAuuq7FLHQJjACoYkDvlxychKwiiBIUjcHICiiKoGjaJ54HGKY5gfg4vVxPqkBLJgKTFCwjV6l1NVZB4LDeL4bRGOEMzlJURiFJkIjjEEEH/bd/Q/UMEGdLdPRjO9eRQ49XQsLD0xlIMSRQ1MQNGBevTgxjMhLCowobcxuAbBKA6mCQC0kJ6HY%2BQ2TbWTqkl2NJuA/rFI78lJDg7ApE5ihBOyqRFw6adpDl6QZy5yhqJlmeqipbp5Op7oaPn%2BWgKwkCkrUOneQUS%2BQL5cLw7OCjF/6AcBLpuhBK3iJI0ibc7O1aNuB0DgNKTtaujYedqvD5a1BvgpgdA7FbnPc3JEDi%2BpSlad5ukLrgS5GYHysbmrIe2fqWu6b5yCnvxtroOQmDfCkRvhep5tvidX5c7JAGhPboEQQdJOBtBuyweGCFXDGdzxmhrwYSmPzYbggK4Vm%2BG5giJGomRWI4pRFa0eS9E0oxaKMuTrFuhNJks9xvB8ReOwALJsFXmCC5oKSmBkIiif2Q4XLHNit/HcEo57YpxUibZOUs05zgzlnWgOd1yWXzq2Xcdli7QKciAAAbK5Xg7kkFeR0ug%2BUVI0T/DFGKNErIACcaIqF8D4FYKwaITJWHXHwHQmor62RlqXWAfl%2BFlxAM1dArUKBGyaDNdQpgupiBQJgAcS1BANxMLdKRURGCyPkdZZRQxopt1ipQHR5Aw7iDkQo7cwjgTkBmig4RDRiCckEM7Nabt5AezUF7fa7ojoYBOsYPqF1jLXVuvdSCQYsYOLUTIsxiiT7tUoL7Ng/tz7M2Dsgng%2BBNAtUUn/AB7deZ%2BMFiAyBhDHJy2zhg1WDZc6INZigoupSjx%2BTQMAdgJA8BxD8MAUaFN67gJCk3S2LcbYdyAuOB2YFcak2DIPMM8FIyjxuOPVCzwp7Ji%2BLPP488Mx4RzIRVeyJ14li3uWailY94BAYiCJiPTT4BwvmkmyABxaIwI9g7HpLgTpwB9gsS/uJX%2Bwz9EJyFhpEWYDgqSxFtLQ8MDDJwKVggqyXDNaNN4SeEA14%2BmQrNq%2BIZ/M8mxTtuM7ulBe7TK2LMuCEYLiLOQhPVZSZMKbJwpmcEy99n5kOcWciJyqI0RJBcykB9rlHxnCxNiAhA6X3VrwMIYQex3x2MJf5P8Y5AsAfJYp4Kxb9KhVOKBZTM7wuMtUpF%2BCNaoLRTrEAKA2AqFlF1RgPg4y3iMYMnguSRnEudKS8l/cZmhmpSPaMSy4wrPQusrCWyF5suzAROEBzCxHN5dibeZzd7VhFXSMVJ9JUcSDha3gTyBJeF1XiVVEkNX5OAeM0BuqcUpxhenfSxr5aIvMnnOphd7KwsqTgngeDu2GqacgZpIBgnYtNh6r1wLRld0dmSqZAbKVBuHgs0N9KI1rOZWmbZi92V7MTVy5NPLN5ptOQKjgdFLnZpuRKs%2BUqHlFp4HTStgKCUjK1XWzSELTZNpHXC9t8oqlrk7bUlFVqeE2taWwdpXyulTsbniz11aiWdxJYu/1UFA1D3mbSzdyyEw7pnnu2NuyE1EQpKRY5F7%2BXnKzbWXNtz83SseTqF5byPkIZ%2BXNKtn653fsUr%2Bht/6SkyyAxUs14HkWyp7Wgk0NrGCMFqhwOm6giDHF6lEOaRseAAFo6AsEwPp4gKhJDkGwPp2qUg/C4H0%2BwLgKgeDbndSh2dgDBBlSOswfTKgDgVF8I1bk2AAgpEM3%2BRg%2BmOA4D/JTH1IFF0MH/IkfTkRLRqGhKwVwUh9NvxUGoRqy1FAuI2m4xQns9rcjQMp1T6nNNdRmLp3uSNErYAAGJ%2BG7LdeprWgIMzwNoJUGQOlNAAOqukvBuQ0D77mpJffKxVOw30aa0zMd96qBOatrcJnVScQoAcaZJhF0mLKyYLqi6D467UOqdQ0YASGBnubQ7bDDvqsPLpw6uvDNLEJj3DcRplpG57kaXkeqja8z3Agopehj%2B8mPHxY4%2BgtMqC4lrLWIPYTyNseZraC%2Bt%2B39WpyO622BpqwNnZfZdvtIGB1DsgxJypsh/h8EwWz9nHP2fyAbKwzt7DOFyZHeim1KQ65uo9MAQznY4PxLGe9owyXErkH0%2BU4LTiSuuzK1tZQHiqsGCMEjubnEX3qDxCkau6wCtPBMRHTbHN/5fp28LZSomIHQsA6Tk1CsrBUJZ5z/3bOAAcHbKfDqg9rcd6A6APbdXq3FFtUNbfbgliZPcjDBKyMldgtUiDADg89diBQUAHAyOgIrJpCi0AAPo16G5QGvVeso6GJpr3GfWOucg5cekgCxeB5uR2xl9Ty/YoBxy9oB%2BOROE7BQakn5STsU67ZBhpV3BF4Bz/aW86%2BiBBlsnLxLCuNfrRkOV7auvvZt%2BcD1iAngcYQS%2BujOYmMCjpABtkeGwMX9FCyATJ/l%2BnrdDYwf7/7Iyoy/6/RYy9B34jD4zfSExaT95G6FrdqfIb4nB8Yfr26Eograou7T6HaM7z7k6mTmph4r4R6CI3aOrKZeAcAcCPbPjPZJ7ob76p5LpQwUowRzK/Z0pEaTxA4bJkY7Jg6UZJo0applj0aZrw6HyI6zYpLG7dro5ix0Fj7ME4E/p7Zx4EF9pEHwIyZU7h4tryhWCB4s5WD0KB6yAkKWHWFohUgmT06C7WoCLmggD2pxSx44ozrj4p5%2Bqfa75cHBobpIR8GMrTyCEg7CGHqiEnriHnqSE7yCqMayHioUysbPrdrAi3wYF27WyCZO4z6iz4Hia6FtpSaL6qxkG9olw2pngXgcBYpeHTpMFYHepvYH5p4cErpBHroEahEA78ERHRqsoUYrxxEpoJHppXo3rCoI5pF3IKHIFcI3zgh3wqq5G448yFEE7aGlEtp6Eh7BCYL/A6BihUh0KB4MKB6YKsiyBULB7VEKajpuHADoDoChYPhGxV5uYJ5bGvY/Fx6kpV6sHAmG6F5vDYB4gFZ0wF4ShSCMCwnuCF5xAvDoCygqBImxK/HNzqEdx4CNBMCYk3Rwl%2BFJZH6uLa6VbbgtLvGfFi5vz0hhYX4QTaasCYBYnxIDTzQknIkSjNCMBMkPjbjQxARSIFYIrQzPQ36vTQEP6zAQFf5v7QGgw/5wF/6IxX71BAG5Cf5SmAGwGP6Kl4xwy6m4xTDgGQzEy9Rqa8mxJ4wkDEk9aF63JCnJLcjgkSh%2BCuicmF61TXROmkmD5KFeBlp7AsSbA8R7BhBqFtEFGT5aGNr7HoKHGnbBBUj/A0Js46BXGB5oh8BUiyCKghBPEk7yh8CkL5kWGyCB5UKYLVm1n1mOHsYoKM7yj1n/BoiYJdlWAuRWA5l5lojc68C87qj86GGM484tncJ9r1TkAfxBCyBAA',
            {
                onBeforeLoad: win => {
                    stubConsoleOutput(win);
                    win.localStorage.clear();
                },
            },
        );
    });

    afterEach('Ensure no output in console', () => {
        cy.window().then(win => {
            assertNoConsoleOutput();
            win.localStorage.clear();
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
