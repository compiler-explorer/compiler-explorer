# Claude Explain Implementation Plan

This document outlines the implementation plan for adding a "Claude Explain" feature to Compiler Explorer. This feature will allow users to get an AI-powered explanation of their code and its compilation results using Claude.

## Overview

The Claude Explain feature is implemented completely as a client-side feature that:
1. Takes compilation results from the client (source code, assembly, etc.)
2. Sends them directly from the client browser to the Claude Explain API
3. Displays the AI-generated explanation to the user

This approach distributes API load across clients rather than funneling through the CE server, making rate limiting more natural and reducing server load.

## Implementation Status

- [x] Create minimal server-side configuration in properties files
- [x] Add tool registration in client-side config
- [x] Implement client-side API integration in the UI
- [x] Add user consent UI before sending data to API
- [x] Implement error handling and loading states
- [x] Update documentation
- [x] Replace Monaco editor with HTML-based markdown rendering
- [x] Test the implementation with various languages

## Technical Implementation Details

### Client-Side Implementation (Completed)

#### API Integration (Completed)

We've implemented direct API calls from the client using `fetch` inside the `ExplainView` class:

- The API endpoint is configured through the `explainApiEndpoint` option (which comes from `compiler-explorer.*.properties`)
- Requests include source code, compiler info, and assembly output
- Responses are displayed as markdown in the tool window
- Error handling is in place for API communication issues

#### UI Implementation (Completed)

1. **Pane View**:
   - Created `ExplainView` class extending the existing `Pane` class
   - Set up proper markdown rendering for the explanation
   - Implemented window title updating
   - Created pug template at `views/templates/panes/explain.pug`
   - Added extensive CSS styling with initial theme support

2. **User Consent**:
   - Added a consent UI that appears before sending data to API
   - Implemented session-persistent consent using a static class variable
   - After consenting once, subsequent compilations explain automatically

3. **Loading State**:
   - Shows a loading indicator during API calls
   - Provides user feedback during explanation generation

4. **Error Handling**:
   - Displays user-friendly error messages
   - Handles network errors and API communication issues

## Current Operation

1. When compilation completes, the Claude Explain tool shows the consent UI
2. User grants consent (stored for the session)
3. Data is sent directly to the Claude Explain API
4. The markdown explanation is displayed in the tool window
5. Subsequent compilations automatically generate new explanations (no need to re-consent)

## Next Steps

1. Test with various languages and compilation scenarios
2. Evaluate API performance and rate limiting needs
3. Refine instruction set and language detection
4. Address UI styling improvements based on user feedback:
   - Update UI styling for "generating explanation" to work with all themes (especially pink theme)
   - Make "generating explanation" part of the toolbar with proper animation and status colors (similar to `.status-icon` in compiler view)
   - Reduce excessive padding on the bottom bar
   - Dark mode AI disclaimer is invisible
   - General toolbar alignment and padding
   - Bottom bar sizing
5. **Update privacy policy** - we need to mention this in the privacy policy, and check the wording on the "Send my code" consent thingy.
6. Proper theming support across all color schemes. we seem to have put the dark theme in the main sccs.
7. Disclaimers about AI limitations and potential inaccuracies. Better and longer text here
8. Be much clearer about what information is sent before the first submission. Folks didn't necessarily know the source was sent too
9. Support a `// no-ai` magic comment in source to _never_ send to Anthropic etc (cc @hanadusikova)
