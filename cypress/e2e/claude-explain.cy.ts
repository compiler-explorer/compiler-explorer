// Copyright (c) 2025, Compiler Explorer Authors
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

import {clearAllIntercepts, setMonacoEditorContent, stubConsoleOutput} from '../support/utils';

// Claude Explain specific test utilities
function mockClaudeExplainAPI() {
    // Mock GET request for options
    cy.intercept('GET', 'http://test.localhost/fake-api/explain', {
        statusCode: 200,
        body: {
            audience: [
                {value: 'test_first', description: 'Test first audience level'},
                {value: 'test_second', description: 'Test second audience level'},
                {value: 'test_third', description: 'Test third audience level'},
            ],
            explanation: [
                {value: 'focus_a', description: 'Test focus A explanation type'},
                {value: 'focus_b', description: 'Test focus B explanation type'},
                {value: 'focus_c', description: 'Test focus C explanation type'},
            ],
        },
    }).as('getOptions');

    // Mock POST request for explanation
    cy.intercept('POST', 'http://test.localhost/fake-api/explain', {
        statusCode: 200,
        body: {
            status: 'success',
            explanation: '## Test Assembly Explanation\nThis is a test explanation from the mocked API.',
            usage: {totalTokens: 150},
            model: 'claude-3-test',
            cost: {totalCost: 0.001},
        },
    }).as('explainRequest');
}

function openClaudeExplainPane() {
    cy.get('[data-cy="new-compiler-dropdown-btn"]:visible').click();
    cy.get('[data-cy="new-view-explain-btn"]:visible').click();
}

function setupClaudeExplainEnvironment() {
    // Start with clean intercepts
    clearAllIntercepts();

    // Belt and suspenders: block production API calls
    cy.intercept('https://api.compiler-explorer.com/**', {statusCode: 500, body: {error: 'BLOCKED PRODUCTION API'}}).as(
        'blockedProduction',
    );

    // Set up configuration
    cy.visit('/', {
        onBeforeLoad: (win: any) => {
            stubConsoleOutput(win);
            win.compilerExplorerOptions = win.compilerExplorerOptions || {};
            win.compilerExplorerOptions.explainApiEndpoint = 'http://test.localhost/fake-api/explain';
        },
    });

    // Force override after page loads
    cy.window().then((win: any) => {
        win.compilerExplorerOptions.explainApiEndpoint = 'http://test.localhost/fake-api/explain';
    });
}

function mockClaudeExplainAPIWithOptions() {
    // Just call the standard mock for consistency
    mockClaudeExplainAPI();
}

function mockSuccessfulExplanation() {
    cy.intercept('POST', 'http://test.localhost/fake-api/explain', {
        statusCode: 200,
        body: {
            status: 'success',
            explanation:
                '## Understanding Your Assembly Code\n\nThis simple program:\n\n```cpp\nint main() {\n    return 42;\n}\n```\n\nCompiles to very efficient assembly that:\n1. Sets up the stack frame\n2. Moves the value 42 into the return register\n3. Cleans up and returns\n\n### Key Instructions\n- `mov eax, 42` - Places our return value in the EAX register\n- `ret` - Returns control to the caller',
            usage: {
                inputTokens: 250,
                outputTokens: 120,
                totalTokens: 370,
            },
            model: 'claude-3-haiku',
            cost: {
                inputCost: 0.00025,
                outputCost: 0.0012,
                totalCost: 0.00145,
            },
        },
    }).as('explainRequest');
}

function mockAPIError() {
    cy.intercept('POST', 'http://test.localhost/fake-api/explain', {
        statusCode: 500,
        body: {error: 'Internal server error'},
    }).as('explainError');
}

function giveConsentAndWait() {
    cy.get('.consent-btn:visible').click();
    cy.wait('@explainRequest');
}

function mockCachedExplanation() {
    const mockResponse = {
        status: 'success',
        explanation: 'This is a cached explanation',
        usage: {totalTokens: 100},
        model: 'claude-3',
        cost: {totalCost: 0.001},
    };

    cy.intercept('POST', 'http://test.localhost/fake-api/explain', mockResponse).as('explainRequest');
}

// Simplified test helpers
function waitForDropdownsToLoad() {
    cy.get('.explain-audience:visible option[value="loading"]').should('not.exist');
    cy.get('.explain-type:visible option[value="loading"]').should('not.exist');
}

function openClaudeExplainPaneWithOptions() {
    mockClaudeExplainAPI();
    openClaudeExplainPane();
    cy.wait('@getOptions');
    waitForDropdownsToLoad();
}

describe('Claude Explain feature', () => {
    beforeEach(() => {
        setupClaudeExplainEnvironment();
    });

    afterEach(() => {
        // Clear all Cypress intercepts to prevent O(n²) accumulation
        cy.state('routes', []);
        cy.state('aliases', {});

        // Clear storage and state but avoid aggressive DOM manipulation
        cy.window().then(win => {
            // Clear any stored explain state/cache
            win.localStorage.clear();
            win.sessionStorage.clear();

            // Reset compiler explorer options
            win.compilerExplorerOptions = {};

            // Instead of force-closing tabs, let natural cleanup happen
            // The next beforeEach will reload the page anyway
        });
    });

    describe('Basic functionality', () => {
        it('should open Claude Explain pane from compiler toolbar', () => {
            openClaudeExplainPaneWithOptions();

            // Verify the explain pane opened
            cy.get('.lm_title').should('contain', 'Claude Explain');
            cy.get('.explain-consent').should('be.visible');
        });

        it('should show consent dialog on first use', () => {
            openClaudeExplainPaneWithOptions();

            // Verify consent dialog is shown (use :visible to avoid template elements)
            cy.get('.explain-consent:visible').should('be.visible');
            cy.get('.explain-consent:visible').should('contain', 'Consent Request');
            cy.get('.explain-consent:visible').should('contain', 'Claude Explain will send your source code');
            cy.get('.explain-consent:visible').should('contain', 'compilation output');
            cy.get('.explain-consent:visible').should('contain', 'Anthropic');

            // Verify consent button exists
            cy.get('.consent-btn:visible').should('be.visible');
            cy.get('.consent-btn:visible').should('contain', 'Yes, explain this code');
        });

        it('should remember consent for the session', () => {
            mockClaudeExplainAPIWithOptions();
            mockClaudeExplainAPI();

            // Open first explain pane and give consent
            openClaudeExplainPane();
            cy.get('.consent-btn:visible').click();
            cy.wait('@explainRequest');

            // Close the explain pane
            cy.get('.lm_close_tab').last().click();

            // Set up mocks again before reopening
            mockClaudeExplainAPIWithOptions();

            // Open a new explain pane
            openClaudeExplainPane();

            // Verify consent dialog is NOT shown again (check for visible consent, not template)
            cy.get('.explain-consent:visible').should('not.exist');
            cy.get('.explain-content').should('be.visible');
        });
    });

    describe('no-ai directive handling', () => {
        it('should detect no-ai directive and show special message', () => {
            // Update code to include no-ai directive
            setMonacoEditorContent('// no-ai\nint main() {\n    return 42;\n}');

            openClaudeExplainPaneWithOptions();

            // Verify no-ai message is shown (use :visible to avoid template elements)
            cy.get('.explain-no-ai:visible').should('be.visible');
            cy.get('.explain-no-ai:visible').should('contain', 'AI Explanation Not Available');
            cy.get('.explain-no-ai:visible').should('contain', 'no-ai');

            // Verify consent dialog is NOT shown (check visible elements only)
            cy.get('.explain-consent:visible').should('not.exist');
        });

        it('should detect case-insensitive no-ai directive', () => {
            // Update code with different case
            setMonacoEditorContent('// NO-AI\nint main() {\n    return 0;\n}');

            openClaudeExplainPaneWithOptions();

            // Should still detect it (use :visible to avoid template elements)
            cy.get('.explain-no-ai:visible').should('be.visible');
        });
    });

    describe('API interaction and explanations', () => {
        it('should fetch and display explanation after consent', () => {
            mockClaudeExplainAPIWithOptions();
            mockSuccessfulExplanation();

            // Open explain pane
            openClaudeExplainPane();

            // Give consent and wait for API call
            giveConsentAndWait();
            cy.wait('@getOptions');

            // Verify explanation is displayed (use :visible to avoid template elements)
            cy.get('.explain-content:visible').should('be.visible');
            cy.get('.explain-content:visible').should('contain', 'Understanding Your Assembly Code');
            cy.get('.explain-content:visible').should('contain', 'mov eax, 42');

            // Verify markdown is rendered (should have headers)
            cy.get('.explain-content:visible h2').should('exist');
            cy.get('.explain-content:visible h3').should('exist');
            cy.get('.explain-content:visible code').should('exist');

            // Verify stats are shown
            cy.get('.explain-stats:visible').should('be.visible');
            cy.get('.explain-stats:visible').should('contain', 'Fresh');
            cy.get('.explain-stats:visible').should('contain', 'Model: claude-3-haiku');
            cy.get('.explain-stats:visible').should('contain', 'Tokens: 370');
            cy.get('.explain-stats:visible').should('contain', 'Cost: $0.001450');
        });

        it('should handle API errors gracefully', () => {
            mockClaudeExplainAPIWithOptions();
            mockAPIError();

            // Open pane and consent
            openClaudeExplainPane();
            cy.get('.consent-btn:visible').click();

            // Wait for error
            cy.wait('@explainError');

            // Verify error is displayed (use :visible to avoid template elements)
            cy.get('.explain-content:visible').should('contain', 'Error');
            cy.get('.explain-content:visible').should('contain', 'Server returned 500');

            // Verify error icon
            cy.get('.status-icon.fa-times-circle:visible').should('be.visible');
        });

        it('should show loading state during API call', () => {
            mockClaudeExplainAPIWithOptions();

            // Mock slow API response with direct reply (no network request)
            cy.intercept('POST', 'http://test.localhost/fake-api/explain', {
                delay: 1000,
                statusCode: 200,
                body: {
                    status: 'success',
                    explanation: 'Test explanation',
                },
            }).as('slowExplain');

            // Open pane and consent
            openClaudeExplainPane();
            cy.get('.consent-btn:visible').click();

            // Verify loading state (use :visible to avoid template elements)
            cy.get('.status-icon.fa-spinner.fa-spin:visible').should('be.visible');
            cy.get('.explain-content:visible').should('contain', 'Generating explanation...');

            // Wait for response
            cy.wait('@slowExplain');

            // Verify loading state is gone
            cy.get('.status-icon.fa-spinner:visible').should('not.exist');
            cy.get('.status-icon.fa-check-circle:visible').should('be.visible');
        });
    });

    describe('Options and customization', () => {
        it('should load and display audience and explanation options', () => {
            openClaudeExplainPaneWithOptions();

            // Verify audience dropdown (use :visible to avoid template elements)
            cy.get('.explain-audience:visible').should('be.visible');
            cy.get('.explain-audience:visible option').should('have.length', 3);
            cy.get('.explain-audience:visible option[value="test_first"]').should('exist');
            cy.get('.explain-audience:visible option[value="test_second"]').should('exist');
            cy.get('.explain-audience:visible option[value="test_third"]').should('exist');

            // Verify explanation type dropdown
            cy.get('.explain-type:visible').should('be.visible');
            cy.get('.explain-type:visible option').should('have.length', 3);
            cy.get('.explain-type:visible option[value="focus_a"]').should('exist');
            cy.get('.explain-type:visible option[value="focus_b"]').should('exist');
            cy.get('.explain-type:visible option[value="focus_c"]').should('exist');

            // Verify info buttons
            cy.get('.explain-audience-info:visible').should('be.visible');
            cy.get('.explain-type-info:visible').should('be.visible');
        });

        it('should show popover descriptions for options', () => {
            openClaudeExplainPaneWithOptions();

            // Click audience info button (use :visible to avoid template elements)
            cy.get('.explain-audience-info:visible').click();

            // Verify popover appears
            cy.get('.popover:visible').should('be.visible');
            cy.get('.popover-body:visible').should('contain', 'Test_first:');
            cy.get('.popover-body:visible').should('contain', 'Test first audience level');

            // Click away to close
            cy.get('.explain-content:visible').click();
            cy.get('.popover:visible').should('not.exist');

            // Click explanation info button
            cy.get('.explain-type-info:visible').click();
            cy.get('.popover:visible').should('be.visible');
            cy.get('.popover-body:visible').should('contain', 'Focus_a:');
            cy.get('.popover-body:visible').should('contain', 'Test focus A');
        });

        it('should re-fetch explanation when options change', () => {
            // Block production API
            cy.intercept('https://api.compiler-explorer.com/**', {
                statusCode: 500,
                body: {error: 'BLOCKED PRODUCTION API'},
            }).as('blockedProduction');

            // Only set up GET mock for options, not POST (we have custom POST below)
            cy.intercept('GET', 'http://test.localhost/fake-api/explain', {
                statusCode: 200,
                body: {
                    audience: [
                        {value: 'test_first', description: 'Test first audience level'},
                        {value: 'test_second', description: 'Test second audience level'},
                        {value: 'test_third', description: 'Test third audience level'},
                    ],
                    explanation: [
                        {value: 'focus_a', description: 'Test focus A explanation type'},
                        {value: 'focus_b', description: 'Test focus B explanation type'},
                        {value: 'focus_c', description: 'Test focus C explanation type'},
                    ],
                },
            }).as('getOptions');

            let explainCallCount = 0;
            cy.intercept('POST', 'http://test.localhost/fake-api/explain', (req: Cypress.Interception) => {
                explainCallCount++;
                req.reply({
                    status: 'success',
                    explanation: `Explanation #${explainCallCount} for ${req.body.audience} audience`,
                    usage: {totalTokens: 100},
                    model: 'test-model',
                    cost: {totalCost: 0.001},
                });
            }).as('explainRequest');

            // Open pane and wait for options to load first
            openClaudeExplainPane();
            cy.wait('@getOptions');
            waitForDropdownsToLoad();

            // Give consent
            giveConsentAndWait();

            // Wait a bit for content to render
            cy.wait(100);

            // Verify initial explanation (just check for the count, not the full text)
            cy.get('.explain-content:visible').should('contain', 'Explanation #1');

            // Change audience level
            cy.get('.explain-audience:visible').select('test_second');

            // Should trigger new request
            cy.wait('@explainRequest');
            cy.get('.explain-content:visible').should('contain', 'Explanation #2');

            // Change explanation type
            cy.get('.explain-type:visible').select('focus_b');

            // Should trigger another request
            cy.wait('@explainRequest');
            cy.get('.explain-content:visible').should('contain', 'Explanation #3');
        });
    });

    describe('Caching and reload', () => {
        it('should cache responses and show cache status', () => {
            mockClaudeExplainAPIWithOptions();
            mockCachedExplanation();

            // Open pane and get first explanation
            openClaudeExplainPane();
            giveConsentAndWait();

            // Verify fresh status (use :visible to avoid template elements)
            cy.get('.explain-stats:visible').should('contain', 'Fresh');

            // Close and reopen pane (should use client cache)
            cy.get('.lm_close_tab').last().click();

            // Set up options mock only (needed for pane constructor) but NOT explanation mock
            mockClaudeExplainAPIWithOptions();
            openClaudeExplainPane();

            // Since consent was already given, it should go straight to cached content
            // (no consent dialog should appear)
            cy.get('.explain-consent:visible').should('not.exist');

            // Should use cached explanation data
            cy.get('.explain-content:visible').should('contain', 'This is a cached explanation');
            cy.get('.explain-stats:visible').should('contain', 'Cached (client)');
        });

        it('should bypass cache when reload button is clicked', () => {
            mockClaudeExplainAPIWithOptions();

            let callCount = 0;
            cy.intercept('POST', 'http://test.localhost/fake-api/explain', (req: Cypress.Interception) => {
                callCount++;
                const isBypassCache = req.body.bypassCache === true;
                req.reply({
                    status: 'success',
                    explanation: `Explanation #${callCount}${isBypassCache ? ' (bypassed cache)' : ''}`,
                });
            }).as('explainRequest');

            // Get initial explanation
            openClaudeExplainPane();
            giveConsentAndWait();

            cy.get('.explain-content:visible').should('contain', 'Explanation #1');

            // Click reload button (use :visible to avoid template elements)
            cy.get('.explain-reload:visible').click();
            cy.wait('@explainRequest');

            // Should have made a new request with bypassCache flag
            cy.get('.explain-content:visible').should('contain', 'Explanation #2 (bypassed cache)');
        });
    });

    describe('Compilation state handling', () => {
        it('should handle compilation failures', () => {
            // Add invalid code
            setMonacoEditorContent('this is not valid C++ code');

            openClaudeExplainPaneWithOptions();

            // Should show compilation failed message (use :visible to avoid template elements)
            cy.get('.explain-content:visible').should('contain', 'Cannot explain: Compilation failed');

            // Should not show consent dialog
            cy.get('.explain-consent:visible').should('not.exist');
        });

        it('should update explanation when code changes', () => {
            mockClaudeExplainAPIWithOptions();

            let explainCount = 0;
            cy.intercept('POST', 'http://test.localhost/fake-api/explain', (req: Cypress.Interception) => {
                explainCount++;
                req.reply({
                    status: 'success',
                    explanation: `Explanation for version ${explainCount}`,
                });
            }).as('explainRequest');

            // Open pane and get initial explanation
            openClaudeExplainPane();
            giveConsentAndWait();

            cy.get('.explain-content:visible').should('contain', 'Explanation for version 1');

            // Change the code
            setMonacoEditorContent('int main() {\n    return 100;\n}');

            // Wait for new explanation
            cy.wait('@explainRequest');

            cy.get('.explain-content:visible').should('contain', 'Explanation for version 2');
        });
    });

    describe('UI state and theming', () => {
        it('should persist option selections in state', () => {
            // Mock API with test options
            mockClaudeExplainAPI();

            // Open pane
            openClaudeExplainPane();

            // Add error listener to catch JS errors
            cy.window().then(win => {
                win.addEventListener('error', e => {
                    cy.log('JS Error:', e.message, 'at', e.filename + ':' + e.lineno);
                });
                win.addEventListener('unhandledrejection', e => {
                    cy.log('Unhandled Promise Rejection:', e.reason);
                });
            });

            // Wait for options to load
            cy.wait('@getOptions');

            // Wait for dropdown to be populated (wait for "Loading..." to disappear)
            cy.get('.explain-audience:visible option[value="loading"]').should('not.exist');
            cy.get('.explain-type:visible option[value="loading"]').should('not.exist');

            // Give consent first to make dropdowns available
            cy.get('.consent-btn:visible').click();

            // Wait for dropdowns to be populated with actual options
            cy.get('.explain-audience:visible option[value="test_second"]').should('exist');
            cy.get('.explain-type:visible option[value="focus_b"]').should('exist');

            // Now change options after consent (use :visible to avoid template elements)
            cy.get('.explain-audience:visible').select('test_second');
            cy.get('.explain-type:visible').select('focus_b');

            // Wait for the explanation request triggered by option changes
            cy.wait('@explainRequest');

            // Get the current URL (which includes state)
            cy.url().then((url: string) => {
                // Clear intercepts from previous test
                cy.state('routes', []);
                cy.state('aliases', {});

                // Set up mocks BEFORE visiting
                mockClaudeExplainAPI();

                // Block production API
                cy.intercept('https://api.compiler-explorer.com/**', {
                    statusCode: 500,
                    body: {error: 'BLOCKED PRODUCTION API'},
                }).as('blockedProduction');

                // Visit the URL with configuration
                cy.visit(url, {
                    onBeforeLoad: (win: any) => {
                        win.compilerExplorerOptions = win.compilerExplorerOptions || {};
                        win.compilerExplorerOptions.explainApiEndpoint = 'http://test.localhost/fake-api/explain';
                    },
                });

                // Force override after page loads (in case it gets reset)
                cy.window().then((win: any) => {
                    win.compilerExplorerOptions = win.compilerExplorerOptions || {};
                    win.compilerExplorerOptions.explainApiEndpoint = 'http://test.localhost/fake-api/explain';
                });

                // Verify options are restored
                cy.get('.explain-audience:visible').should('have.value', 'test_second');
                cy.get('.explain-type:visible').should('have.value', 'focus_b');
            });
        });
    });
});
