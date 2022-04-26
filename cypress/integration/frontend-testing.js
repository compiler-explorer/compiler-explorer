function runFrontendTest(name) {
    it(name, () => {
        cy.window().then(win => {
            return win.compilerExplorerFrontendTesting.run(name);
        });
    });
}

const click = $el => $el.click();

describe.skip('Frontendtestresults', () => {
    before(() => {
        cy.visit('/');
    });

    runFrontendTest('HelloWorld');
});

describe('Prod happy path', () => {
    before(() => {
        // cy.visit('https://godbolt.org');
        cy.visit('/', {
            onBeforeLoad(win) {
                cy.stub(win.console, 'log').as('consoleLog');
                cy.stub(win.console, 'error').as('consoleError');
            },
        });
    });

    it('Works on a happy path', () => {
        // Initially, nothing is set, so first thing we do is ensure the policy modal showed up
        policiesHappyPath();

        // Now, we are ready to check the panes!
        editorHappyPath();
        compilerHappyPath();

        executionHappyPath();

        // Finally, none of this should have printed anything to the console
        cy.get('@consoleLog').should('not.be.called');
        cy.get('@consoleError').should('not.be.called');
    });
});

function policiesHappyPath() {
    cy.get('#alert').within(() => {
        cy.get('.modal-title').contains('New Privacy Policy. Please take a moment to read it').should('be.visible');
        cy.get('.modal-footer button')
            .pipe(click)
            .should($el => expect($el).to.not.be.visible);
    });

    // Now, same for the cookies popup
    cy.get('#simplecook')
        .should('be.visible')
        .within(() => {
            cy.get('button').should('have.length', 3); // 1 to open the modal, 2 to (not) consent
            cy.get('button.cookies').first().click();
        });

    cy.get('#yes-no')
        .should('be.visible')
        .within(() => {
            cy.get('.modal-title').should('include.text', 'Denied');
            cy.get('.modal-footer button').should('have.length', 2);

            // Close the modal, we'll test it later
            cy.get('.modal-header button')
                .pipe(click)
                .should($el => expect($el).to.not.be.visible);
        });

    // Grant this monster a cookie for breakfast
    cy.get('#simplecook button.cook-do-consent').click();

    // Check that the monster ate the cookie
    cy.get('button#policiesDropdown').should('be.visible').click();
    cy.get('button#cookies').click();
    cy.get('#yes-no')
        .should('be.visible')
        .within(() => {
            cy.get('.modal-title').should('include.text', 'Granted');
            // Give the cookie back
            cy.get('.modal-footer button.no')
                .pipe(click)
                .should($el => expect($el).to.not.be.visible);
        });

    // Check that the monster returned the cookie
    cy.get('button#policiesDropdown').click();
    cy.get('button#cookies').click();
    cy.get('#yes-no')
        .should('be.visible')
        .within(() => {
            cy.get('.modal-title').should('include.text', 'Denied');
            cy.get('.modal-footer button.no')
                .pipe(click)
                .should($el => expect($el).to.not.be.visible);
        });
}

function editorHappyPath() {
    // The default config has 2 items: Editor & Compiler
    cy.get('.lm_content')
        .should('have.length', 2)
        .first()
        .within(() => {});
}

function compilerHappyPath() {
    // The default config has 2 items: Editor & Compiler
    cy.get('.lm_item.lm_stack').should('have.length', 2).eq(1).as('compilerStack');

    // First, test the (re)naming
    cy.get('@compilerStack').within(() => {
        // Default compiler ids is editor 1, compiler 1
        cy.get('.lm_title').should('include.text', 'Editor #1, Compiler #1');
        cy.get('.lm_modify_tab_title').click();
    });

    cy.get('#enter-something')
        .should('be.visible')
        .within(() => {
            cy.get('.modal-title').contains('Rename pane').should('exist');
            cy.get('input.question-answer')
                .should('include.value', 'Editor #1, Compiler #1')
                .clear()
                .type('Cypress Compiler #1');
            cy.get('.modal-footer button.yes')
                .pipe(click)
                .should($el => expect($el).to.not.be.visible);
        });
    cy.get('@compilerStack').within(() => {
        cy.get('.lm_title').should('contain.text', 'Cypress Compiler #1');
    });
}

function executionHappyPath() {
    // The default config has 2 items: Editor & Compiler
    cy.get('.lm_content')
        .should('have.length', 2)
        .first()
        .within(() => {
            cy.get('button.add-pane').click();
            cy.get('.dropdown-menu button.add-executor').click();
        });
}
