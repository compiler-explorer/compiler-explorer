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

describe('Known good state test', () => {
    beforeEach(() => {
        cy.visit(
            // This URL manually created. If you need to update, run a local server like the docs/UsingCypress.md say,
            // then paste this into your browser, make the changes, and then re-share as a full link.
            'http://localhost:10240/#z:OYLghAFBqd5TB8IAsQGMD2ATApgUWwEsAXTAJwBoiQIAzIgG1wDsBDAW1xAHIBGHpTqYWJAMro2zEHwAsQkSQCqAZ1wAFAB68ADIIBWMyozYtQ6AKQAmAELWblNc3QkiI2q2wBhTIwCuHCwgVpSeADJELLgAcgEARrjkIPIADpgqpG4sPv6BwZRpGa4iEVGxHAlJ8k64LlliJGzkJDkBQSE1dSINTSSlMfGJyY6Nza15HaN9kQMVQ7IAlI6YfuTo3DwA9JsA1AAqAJ4puDsHK%2BQ7WHg7KIm4lDsUO4yYbNg7pju4mpwpzAB0Fh0AEFIiQdioAI5%2BJq4CBgnYsAILHYWADsdhBO2xO3IuBIqxYiICOwAVMSOBYAMyY4HogAiPCWjF4AFZBEEeHpKJheF57PYIed1qirFSBJQSLomUsANYgVk6Yy8WSCDgKpWc7m8niCFQgJVSrlMyhwWBoLAsYTkDimdbUDzEMjkIjYIwmMwASTdlls9mWq3WvG2%2ByOJzOq0uOBOtzxDyeLzeHyJ31%2BAKBoNEEOhsPhWaRHBR6NpONx%2BMJFLJFOptIZJpZPHZlC1gh1PitFFtLBFADUiLgAO6JHYQQikJ7WcULQRGvQLJa3N5DCDMlVq4Ks5vSnm8PUGyXSpZmqBoch%2BFQkFBEKg0CBYDgpJiJaKcDbAAW2HZ4OhsPyMEiCLgToUK6RjCKIEhSNwcgKKIqgaNoxqUIYfCOLgzjFEEECeOMQSoeEMzlJURiFJkIi4SR6RkSw/REUMqGdJhPRjL4bRGIx9RTLRgxJAxUwUXxvTcXMvFLCQeK4KBOiro2HLbjqABK57ggAEp6Ck7H2g7Du%2Bfqft%2Bv7/iOY7OqK4oPD4D5Phck58NOB7GvOlCLngSQrsqPCqpQ6pWJuLY7rqjj7rOMqUPKshKg2VJyUhOozoepqIMeqAYJgVnMDetD3o%2BGUgMABl/iQQhMCQiT6hAcTbnEkRNAcvAStV7DkAcADycT6LURoSveXCiC1LCMHVSF4HEfjAF4UiMPq/CCHgXbANIw3Xp1RAAG64NN3LfLUfilfVghguh26MEQcTkLVPh4Nu4lEOqM2UOt5BxOkuD0rg80nWYCV0CYwAqFpA4tccnISuB4iSNIMFg/BWjbihximOYH4OCdcT6pASyYCkmHTa2j0ung6PuRx7jYSw3isXk%2BHk8JxGoaRmECQUVGYbT9FoRhnG9EzJMsMx0xlDx7H8ZTeEjEJhFC3ZAZrNBlADqYJBAyQnodglDZNv5Oo7Lpdj6bgP6FcZwE2WKdnxY5C64EublyhqHleT5fnybuQWGglyVoBebDoLK55sMA9y3tl1kvlwvC64KBX/oBJugahYOQZD8jQ2osNIfDvMeOTTMEYLImUUUWRMwzWRs7xHMrd0Iu5GLvP8%2BXwvc6LTfNI30vibgklutJHmay7PA9L7OyqAHJwAzpyNfgbhngqOJtmahOyWTlw62fZIVOS5y4yY7G5brFrv6u7jmJcgaArCQKS7Q6d5pav5Bh2%2BU/RwBoRx26CeKEn0Ep4oMOIW5PDAc50Uj7Rkv3Q%2BPAWq7WvuCTAdAdYvxnkbCAK9rKLw3oeK2NtaC73XL5A%2B2oj7BQ9klU8ylbToHIJgb4KRb4hwyk/COyDDYx3fuOAmYFv4Q1/rBZQadAEGHYuhKuWEcIt2ptgduzMi7kUkbI6iMj641zYgxURXQ%2BZcUlgXQSLFa6twFrMOmYkJJSQgTFYhPAlIXh2AAWR9jQnY%2BBNApFMBkEQml%2BxDguJHfWbC54mQnGbCy98MHrwtnOHBrk8EOwIc7KBe4T5RLCiAAAbJFXg0UiGtldpvO2VI0T/DFGKNErIACcaJyl8D4FYKwaIPJWHXHwHQmoB6RJNMlJK3Tz4gG2ugXaFBb5ND%2BuoUw6ExAoEwAOEGghGFsEwmMqIjBJnTP8owoY%2BUUH/koBsx%2Br5XZ7JgeIKZMztz9OBOQP6rt%2BkNGIJyQQideEyD/nBQRcN3SIwwMjYwp0iaY2xlkXGPJ8aug2vAJYZ5RC3VwHsTAvh/kyyDHxe5SyJmnNmZKCS%2B15agPAX3SxuSeAuJ2qZPxNhp4BJHPyPSDhnELwiQ5FJ29bapMVHE7y%2B8tYkOSZ08heV2AkBGmNTuQcsphKYQcng5LKWz1jpw%2BO/Cf4vP4QAj56jOakwkQYqRMjS7yJ1Yo1mOi6aV00fzHmGimLaPzqa72%2Bi1Hizbia%2Bipiu7mIJTkgKABxaIwI9g7DeqNYA%2BwJJeO0r41hs9jacMXqE9Ka8zZYMts5a2MT3INj3oQ7lgVj5Mr5b068DCJXPilTK1%2B8rnSKszsKDYIZDjHFOOcKM1xYz3EeBcRM7xPipgfOmEECIoQwjxHmcEBYiwYgzKWPEBJyBEgLFWAsNYMx1ixe6t0M0LFep1GEMIPY7E7HUuGnxSDaWytQUE025ll4lqvebfNW8007w5U7bdPKH1n3NCAFAbAVCynQowHwcZg63uYdKqNhVK0gU/kq55UN/7vIziIzV4ic4KLzsY9m%2BrsgKOw8oq1XMHVUzNdaiWtr2b2paAoyj7c3XdwNFunN3qVJeBvXiY9k8z2vxjaZWy8aH6YI6Y%2B3BGa1ycuze0t2D67bsqioSgKQnP1IC/VjIqIGE37PDuBrj2y35AQVTBhg/5EgAFpTBSAOBkaaoMSqmY4EQFQagN3ciM6VcgJm3GOaAjip5UEVWpwQuqkjWRs4U0NRhuiFdsMlxZmXF1FcVHN0NYl515GEuqOIzR%2BLMg6OgXuiAtgYDN2epzcrDjkadNUvnrGvjN6NOCek6mkTMnMmeXiW%2B3NpDT6ewFQs4VwBi0abA%2BW3TUGuFfwgnB15AjAtIY1WI0LucabZfprFg1jq8MreC9XJLjqUtGMi4YnmNrMOiTXfR3uGt5M6l9f6wNuBg3ldPXrCl3Hqu8ZCXVgTjL8lNfTfg8TCSrFJI/T1xgjBVocGVuoIgxxPq4ABrfHgJm6AsEwCZ4gKhJDkGwCZ1aUg/C4BM%2BwLgKgeDbj2cNiD7C5qI2YCZlQBwKgIu3NgAIKQUd/kYCZjgOA/wbAlPpqthnbPuciJaNQ0JWCuCkB539ahrOPJ4X5%2BDbzZvcjQODyH0PYfoRmIjzOVqgIADE/DdkwkfI32BVZ4G0EqDII0mgAHVXSXg3IaMxG6BCMYHmVmHcOZhPZG1Vy9cavvhKTUJ6Jz7M3tZzSDzeSmUo/r/QBhoA31MPyp5VuVHDhfcMmyr6baq5vbdQ2Fx1EWpZGuLrhtbNEtv7ctShhujeMt1xO4djunuGMlYHsx1jYg9jeqD9TwJDLPvoIyg137LLYmx8Bx1hP2C2WteyfHxr8pWSyH%2BHwNJ%2B%2BD%2BH4P/IBsTTOUtLaYk0HPSv0pHocHD0wAUedgWTioX0GwKi5MyyzaSvC/J1VUQyASMFFTy29z7ygXUDxBSBoXWEcyeGOTgWeyjl0x42CWvSn0TSnCjz%2B2fXlCsHKV3yP2IP3wAA4X0uVJM81E8et0A6AM9xUhsy0x8xtq1kNOoTczdNFeAh1cwwQFheBzsPUrsOsvBjcR9g9o13t0Cl5MC71k1mUn1WUF9X0N9qCyFek8BVoiB7RbwtCdCNhWDDNlcACAt05gD5tuCyYK9iMq9dEa91tiNNs0t2DzV29XDSNUtTsjtqNO8pZcsvd6w2RrteA3ptDexvFOMXtz0jJpC71%2BMI9sDGs59RM2tF81Cus5wk9%2BUU9/1wcvAOAOBBss9mCc9IM88P8JtwYi9AD1dhFLCLdrCltpEttos685EG8XCGjCMqNksCMdsvCu8nVei9s/DdEAje8RCmMWNl5CjR8yjYjQ9as5CZ8V8UiWsKCJMr91Dut%2BU6ETBIhijQ5SjoiK0KjxtkIQCe9LtgjRCTA/BrgXE/g2BIhkD/EpCljJ9b1ViU11jV8OUMjeUnJT849JNft8DSDd8rAalSDZBCkYS4S0QqQPJ18wSNDYB%2BVf01NGCSitNJDyj38Lia1Aw61dgG1wxm0rgYw7h4xO1Xhu0Uwfg%2B1cBAQB0sxeCR0ERx1URJ0sQcQZ0KwF1yQl0aQV00RGQhDAifcoFgRbEJ4KtTjUC4iw8Vifs1ilD58xN1R2VATr9ekzwLx7NMo74mC8SWDzjFVfNTCEM6jLjuitU0NwtlsuiHCcNDVnDvD7StFdtMt%2BjvTBjq9KNjsyNPSJibjZIOsbFwQ7Ej15S3jXslTPiMDvi1TfiNTUis00l/gdAxQqRqlSDalSC0lt9ylyDdSaD%2BVgB0B0A2cHxhlsBsA8RHNlYQt7pJoWz3B7o4gXhfYVAOzMUepTBsAAB9GAg2IgbQdySnE4lAgJWafEF4qafst/D%2BT/ZgEnXAYclQPwOgBgScpYK0vhMwoRSgYAW6P4XAY3XAZc%2B6WHNgekdnILeHNHG8iUDc2Jc6QGQFTsiUZoRgB8h8bcRiICMZRzWJLOJo9DZ0z010mLDo/DFvdwr01vF0oM3wkMrvJYFQE6KHb8zFb2EgPsvCnFUVACorbkUAnuSgPwV0V8wQVaLGIii3YrKY/vLwVjPYCSTYBSPYMIeYxUkPCfZM%2BrVMxQ5rTYqkf4SpffHQAs0gtEPgKkWQRUEIcslfeUPgIpBS6E2QUg8pNJHSvSgylEkIwKcE9JKwf4NENJayqwCKKwWS%2BStEE/XgM/dUC/JfRrEEzy37R6DxIIWQIAA%3D%3D',
            {
                onBeforeLoad: win => {
                    stubConsoleOutput(win);
                },
            },
        );
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
