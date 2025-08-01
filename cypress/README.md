# Cypress E2E Tests

This directory contains end-to-end tests for Compiler Explorer using Cypress.

## Running Tests

### Starting Compiler Explorer for Testing

First, start a local Compiler Explorer instance with a clean configuration:

```bash
npm run dev -- --language c++ --no-local
```

The `--no-local` flag is important as it ensures your setup is clean of any local properties.

### Running Cypress Tests

In another terminal:

```bash
# Run all Cypress tests
npm run cypress

# Run specific test file
npm run cypress -- run --spec "cypress/e2e/claude-explain.cy.ts"

# Open Cypress interactive UI (recommended for development)
npm run cypress:open
```

When using the interactive UI, choose "E2E Testing" and select your browser.

## Important Testing Patterns & Lessons Learned

### 1. **Always Use `:visible` Selectors**
GoldenLayout creates template elements that exist in the DOM but aren't visible. Always use `:visible` to avoid selecting template elements:
```javascript
// ❌ Bad - might select template elements
cy.get('.explain-content').should('contain', 'text');

// ✅ Good - only selects visible elements
cy.get('.explain-content:visible').should('contain', 'text');
```

### 2. **Performance: Clear Intercepts in `afterEach`**
Cypress intercepts accumulate across tests causing O(n²) performance degradation. Always clear them:
```javascript
import {clearAllIntercepts} from '../support/utils';

afterEach(() => {
    // Use the utility function to clear intercepts
    clearAllIntercepts();
    
    // Or manually clear them:
    cy.state('routes', []);
    cy.state('aliases', {});
    
    // ... other cleanup
});
```

### 3. **Mock Setup Timing is Critical**
Always set up API mocks BEFORE any action that might trigger requests:
```javascript
// ❌ Bad - pane constructor might make requests before mocks are ready
openClaudeExplainPane();
mockClaudeExplainAPI();

// ✅ Good - mocks ready before pane opens
mockClaudeExplainAPI();
openClaudeExplainPane();
```

### 4. **Wait for Async DOM Updates**
Don't just wait for API calls - wait for the actual DOM changes:
```javascript
// ❌ Bad - API completes but DOM might not be updated yet
cy.wait('@getOptions');
cy.get('.dropdown').select('value');

// ✅ Good - wait for specific DOM state
cy.wait('@getOptions');
cy.get('.dropdown option[value="loading"]').should('not.exist');
cy.get('.dropdown').select('value');
```

### 5. **Use Test Data, Not Production Values**
Always use clearly fake test data to:
- Prevent confusion with real values
- Make it obvious when viewing test output
- Ensure tests never accidentally hit production APIs

```javascript
// Use values like: test_first, test_second, focus_a, focus_b
// Not: beginner, expert, assembly, optimization
```

### 6. **Helper Functions for Common Patterns**
Extract common test patterns to helpers, but keep them in the test file if they're specific to one feature:
```javascript
// In test file for feature-specific helpers
function openClaudeExplainPaneWithOptions() {
    mockClaudeExplainAPI();
    openClaudeExplainPane();
    cy.wait('@getOptions');
    waitForDropdownsToLoad();
}

// In utils.ts for general helpers
export function setMonacoEditorContent(content: string) { ... }
```

### 7. **State Persistence Between Pane Instances**
Be aware that some state might be static (shared between instances) while other state is per-instance:
- Static state (like consent, cache) persists when closing/reopening panes
- Instance state is lost when panes close
- This affects how you structure tests for features that should persist

### 8. **Block Production APIs**
Always block production API calls in tests to catch configuration issues:
```javascript
cy.intercept('https://api.compiler-explorer.com/**', {
    statusCode: 500, 
    body: {error: 'BLOCKED PRODUCTION API'}
}).as('blockedProduction');
```

## Common Issues & Solutions

### Tests Getting Progressively Slower
- **Cause**: Intercept accumulation
- **Solution**: Clear intercepts in `afterEach` using `clearAllIntercepts()` from utils or manually with `cy.state('routes', [])`

### "Element not found" Despite Being Visible
- **Cause**: Selecting template elements from GoldenLayout
- **Solution**: Use `:visible` pseudo-selector

### API Mocks Not Working
- **Cause**: Mock setup after the request is made
- **Solution**: Set up mocks before opening panes or triggering actions

### Dropdown Selection Failing
- **Cause**: Trying to select before async population completes
- **Solution**: Wait for loading indicators to disappear first

### State Not Persisting in Tests
- **Cause**: Not understanding static vs instance variables
- **Solution**: Check if the feature uses static state that should persist

## Test Organization

- Keep feature-specific test helpers in the test file itself
- Only put truly reusable utilities in `support/utils.ts`
- Use descriptive helper function names that indicate what they do
- Group related tests in `describe` blocks
- Use consistent test data across all tests in a feature

## Debugging Tips

1. Use `cy.log()` to debug what values you're actually getting
2. Check the Cypress command log for unexpected API calls
3. Look for console errors that might indicate JavaScript issues
4. Use `.then()` to inspect element state at specific points
5. Check network tab for requests hitting production instead of mocks