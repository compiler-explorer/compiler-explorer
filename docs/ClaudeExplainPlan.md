# Claude Explain Implementation Plan

This document outlines the implementation plan for adding a "Claude Explain" feature to Compiler Explorer. This feature allows users to get an AI-powered explanation of their code and its compilation results using Claude.

## Overview

The Claude Explain feature has been implemented as a client-side feature that:
1. Takes compilation results from the client (source code, assembly, etc.)
2. Sends them directly from the client browser to the Claude Explain API
3. Displays the AI-generated explanation to the user
4. Caches responses to reduce API costs

This approach distributes API load across clients rather than funneling through the CE server, making rate limiting more natural and reducing server load.

## Implementation Status

- [x] Create minimal server-side configuration in properties files (just API endpoint)
- [x] Implement as standalone pane component (not a tool)
- [x] Add button to compiler toolbar with proper enable/disable logic
- [x] Implement client-side API integration in the UI
- [x] Add user consent UI before sending data to API
- [x] Implement session-persistent consent
- [x] Add LRU caching for API responses
- [x] Implement error handling and loading states
- [x] Add bottom status bar with usage statistics
- [x] Implement markdown rendering with syntax highlighting
- [x] Add reload button to bypass cache
- [x] Test the implementation with various languages
- [x] Update documentation to match implementation

## Technical Implementation Details

### Client-Side Implementation (Completed)

#### Architecture Decision

Implemented as a standalone pane component rather than a tool:
- More direct integration with compiler pane
- Dedicated button in compiler toolbar
- Simpler configuration (no tool registration needed)
- Better UX with immediate access

#### API Integration (Completed)

Implemented direct API calls from the client using `fetch` inside the `ExplainView` class:

- The API endpoint is configured through the `explainApiEndpoint` option
- Requests include source code, compiler name, compilation options, and assembly output
- Responses are cached using LRU cache (200KB limit)
- Responses are displayed as markdown with syntax highlighting
- Error handling is in place for API communication issues

#### UI Implementation (Completed)

1. **Pane View**:
   - Created `ExplainView` class extending the existing `Pane` class
   - Markdown rendering using `marked` library
   - Syntax highlighting with Prism.js
   - Theme-aware styling (light/dark modes)
   - Font scaling support
   - Created pug template at `views/panes/explain.pug`

2. **User Consent**:
   - Clear consent UI that explains what data will be sent
   - Session-persistent consent using a static class variable
   - After consenting once, subsequent compilations explain automatically

3. **Loading State**:
   - Animated loading spinner during API calls
   - Clear "Generating explanation..." message

4. **Error Handling**:
   - User-friendly error messages
   - Network error handling
   - API error handling

5. **Bottom Status Bar**:
   - Shows AI model used
   - Displays token usage (input/output/total)
   - Shows estimated cost
   - Includes reload button to bypass cache

## Current Operation

1. User clicks the "Explain with Claude" button in the compiler toolbar
2. Explain pane opens showing the consent UI (first time only)
3. User grants consent (stored for the session)
4. Data is sent directly to the Claude Explain API (or retrieved from cache)
5. The markdown explanation is displayed with syntax highlighting
6. Bottom bar shows usage statistics and reload button
7. Subsequent compilations automatically update the explanation if the pane is open
8. Users can click reload to bypass cache and get a fresh explanation

## Next Steps

### Immediate - API Integration Updates

#### 1. Fetch and Cache Available Options
- Add method to fetch available options from GET endpoint on first load
- Cache the options for the session (static variable like consent)
- Handle loading state while fetching options
- Gracefully handle if API is unavailable

#### 2. UI State Management for Dynamic Options
**Critical Requirements:**
- Must preserve user's previous choices from saved state/URL even if options haven't loaded yet
- Must handle cases where saved state contains values not in the fetched options
- Must not lose user selections while options are loading

**Implementation Approach:**
```typescript
// Pseudocode for state management
class ExplainView {
  // Static cache for options (shared across all instances)
  private static availableOptions: AvailableOptions | null = null;
  private static optionsFetchPromise: Promise<AvailableOptions> | null = null;

  // Instance state
  private selectedAudience: string = 'beginner';
  private selectedExplanation: string = 'assembly';

  // Load from saved state first, validate later
  loadState(state) {
    this.selectedAudience = state.audience || 'beginner';
    this.selectedExplanation = state.explanation || 'assembly';
    // Don't validate yet - options might not be loaded
  }

  // Fetch options if needed, show UI appropriately
  async ensureOptionsLoaded() {
    if (!ExplainView.availableOptions && !ExplainView.optionsFetchPromise) {
      ExplainView.optionsFetchPromise = this.fetchOptions();
    }
    if (ExplainView.optionsFetchPromise) {
      await ExplainView.optionsFetchPromise;
    }
    // Now validate saved selections against available options
    this.validateSelections();
  }
}
```

#### 3. Add UI Controls
- Add dropdown/select controls for audience level and explanation type
- Position in toolbar or consent area
- Update pug template and styles
- Ensure controls are disabled while options are loading
- Show tooltips with descriptions from the API

#### 4. Update API Request
- Include `audience` and `explanation` parameters in POST request
- Include `bypassCache: true` when reload button is clicked
- Update cache key calculation to include new parameters

#### 5. Display Cache Status
- Parse `cached` field from API response
- Update bottom bar to show cache source:
  - Client cache hit: 🔄 "Cached (client)"
  - Server cache hit: 🔄 "Cached (server)"
  - Fresh generation: ✨ "Fresh"
- Consider adding subtle visual indicator (border color, icon)

### High Priority
1. ~~**Update privacy policy** - mention Claude Explain feature and data handling~~ DONE
2. ~~Support `no-ai` directive in source to prevent sending to Anthropic~~ DONE
3. ~~Fix theme-specific styling issues (especially pink theme)~~ DONE
4. ~~Improve consent UI wording to be clearer about what data is sent~~ DONE
5. ~~Add better disclaimers about AI limitations and potential inaccuracies~~ DONE

### Medium Priority
6. ~~Better language detection based on compiler properties~~ NOT NEEDED - already uses `compiler.lang`
7. ~~Instruction set detection from compiler properties~~ NOT NEEDED - already uses `result.instructionSet`
8. ~~Extract loading state management to base class~~ NOT NEEDED - current approach is clean and other panes use different patterns
9. ~~Add simple frontend test to ensure explain view opens correctly~~ DONE - added to Cypress tests
10. ~~Consider state machine pattern for UI state management~~ OVERKILL - current approach is sufficient
11. ~~Persist user's audience/explanation preferences in localStorage~~ NOT NEEDED

### Low Priority
12. ~~Persistent consent across browser sessions (localStorage)~~ NO - want users to always give consent
13. ~~Improve error handling with different UI states for different error types~~ NOT NEEDED - current error handling is good
14. ~~Consider extracting markdown styles to shared file (200+ lines could be reusable)~~ DONE - extracted to markdown.scss
15. ~~Create explain-view.interfaces.ts file for consistency with other panes~~ DONE
16. ~~Apply theming correctly (pink mode is broken)~~ DONE - themes now work correctly

### Future Enhancements
17. ~~Surface explanation option descriptions in UI (tooltips, TomSelect with descriptions, etc.)~~ DONE - implemented with Bootstrap popovers
18. ~~Consider using richer dropdown component to show both value and description~~ DONE - simple popovers work better than complex dropdowns

### Completed Improvements
- ✅ Session-persistent consent
- ✅ Client-side LRU caching to reduce API costs
- ✅ Bottom status bar with usage statistics
- ✅ Reload button (needs update to bypass server cache too)
- ✅ Loading states with clear messaging
- ✅ Theme-aware styling (though some themes need fixes)
- ✅ Documentation updated to reflect implementation
- ✅ No-AI directive support
- ✅ Bootstrap popover info buttons showing API option descriptions

#### No-AI Directive Implementation
The feature now respects a "no-ai" directive in source code:
- Searches for the string "no-ai" (case-insensitive) in the source code
- If found, displays a polite message explaining why AI explanation is not available
- The message explains that CE respects people who don't want their code processed by AI
- Check happens both when compilation results arrive and before API calls
- Uses a dedicated UI element in the template rather than dynamic HTML

### Implementation Risks & Considerations

1. **Race Conditions**: Must handle case where user triggers explanation before options are fetched
2. **Backwards Compatibility**: Saved states may contain invalid audience/explanation values
3. **Network Failures**: GET request for options might fail - need sensible defaults
4. **Cache Key Changes**: Adding new parameters will invalidate existing cache entries
5. **UI Space**: Need to fit new controls without cluttering the interface

## Detailed Implementation Plan

### Phase 1: API Integration (Backend Communication)

#### 1.1 Update TypeScript Interfaces
Create or update interfaces for the new API structures:
```typescript
interface ExplanationOption {
    value: string;
    description: string;
}

interface AvailableOptions {
    audience: ExplanationOption[];
    explanation: ExplanationOption[];
}

interface ExplainRequest {
    // existing fields...
    audience?: string;
    explanation?: string;
    bypassCache?: boolean;
}

interface ExplainResponse {
    // existing fields...
    cached: boolean;
}
```

#### 1.2 Add Options Fetching
- Add static method to fetch and cache available options
- Implement retry logic with exponential backoff
- Provide hardcoded fallback options if API fails

#### 1.3 Update Cache Key Generation
- Include audience and explanation in cache key
- Ensure consistent key generation

### Phase 2: UI Components

#### 2.1 Update Pug Template
Add select controls to the explain pane.

#### 2.2 Add Styles
- Style the select controls to match CE theme
- Ensure proper spacing and alignment
- Add hover states and transitions

#### 2.3 Wire Up Event Handlers
- Handle selection changes
- Update state and trigger new explanations
- Save preferences to state

### Phase 3: State Management

#### 3.1 Update State Structure
```typescript
interface ExplainViewState {
    // existing fields...
    audience?: string;
    explanation?: string;
}
```

#### 3.2 Implement State Preservation
- Save user selections in component state
- Include in serialized state for sharing
- Restore from saved state properly

### Phase 4: Cache Status Display

#### 4.1 Update Bottom Bar
- Add cache status indicator
- Show different icons/text for different cache sources
- Update dynamically based on response

#### 4.2 Update Reload Behavior
- Send `bypassCache: true` when reload clicked
- Clear client cache for current request
- Show appropriate loading state

### Phase 5: Testing & Polish

#### 5.1 Manual Testing Checklist
- [ ] Options load correctly on first use
- [ ] Selections persist across compilations
- [ ] Saved URLs preserve selections
- [ ] Cache bypass works correctly
- [ ] Cache status displays accurately
- [ ] Graceful handling of API failures
- [ ] UI remains responsive during loading

#### 5.2 Edge Cases to Test
- [ ] API returns unexpected option values
- [ ] Saved state has invalid selections
- [ ] Network failure during options fetch
- [ ] Rapid selection changes
- [ ] Switching between different compilers with explain pane open
