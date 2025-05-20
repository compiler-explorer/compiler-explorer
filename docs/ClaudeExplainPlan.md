# Claude Explain Implementation Plan

This document outlines the implementation plan for adding a "Claude Explain" feature to Compiler Explorer. This feature will allow users to get an AI-powered explanation of their code and its compilation results using Claude.

## Overview

The Claude Explain feature is implemented primarily as a client-side feature that:
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
- [ ] Test the implementation with various languages

## Technical Implementation Details

### 1. Server-Side Configuration (Completed)

Configuration for the tool has been added to the properties files:

```ini
tools.explain.name=Claude Explain
tools.explain.type=postcompilation
tools.explain.stdinHint=disabled
tools.explain.languageId=markdown
tools.explain.class=explain-tool
tools.explain.exe=/bin/true
```

A minimal server-side component (`explain-tool.ts`) was created to provide a placeholder implementation. This doesn't perform any actual execution since all processing happens client-side.

### 2. Client-Side Implementation (Completed)

#### A. Tool Registration

The explain tool has been registered in:
- Property files for each language
- Hub factory method to use the custom `ExplainView` component

#### B. API Integration (Completed)

We've implemented direct API calls from the client using `fetch` inside the `ExplainView` class:

- The API endpoint is configured through the `explainApiEndpoint` option
- Requests include source code, compiler info, and assembly output
- Responses are displayed as markdown in the tool window
- Error handling is in place for API communication issues

#### C. UI Implementation (Completed)

1. **Tool View**:
   - Created `ExplainView` class extending the existing `Tool` class
   - Set up proper markdown rendering for the explanation
   - Implemented window title updating

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

### 3. Cross-Origin Resource Sharing (CORS)

The server API has been configured to support CORS, allowing direct API calls from the client browser to the Claude Explain API. This configuration includes:

- Allowing requests from authorized origins (including localhost for development)
- Supporting the necessary HTTP methods (POST, OPTIONS)
- Including appropriate headers for content type

### 4. HTML-Based Rendering (Completed)

We've replaced the Monaco editor display with a custom HTML-based rendering system:

1. **New ExplainHtmlView Component**:
   - Created a new pug template for the HTML rendering
   - Extended directly from `Pane` rather than from `Tool` or `MonacoPane`
   - Implemented markdown rendering using the `marked` library

2. **Benefits**:
   - Cleaner UI without editor chrome/overhead
   - Better performance (Monaco editor is heavyweight)
   - More flexibility in styling and layout
   - More natural handling of markdown content

3. **Implementation Details**:
   - Created pug template at `views/templates/panes/explain.pug`
   - Created new `ExplainHtmlView` class extending `Pane`
   - Updated hub factory to use the new view class
   - Added extensive CSS styling with theme support
   - Implemented proper font scaling and line wrapping

### 5. Limitations and Future Improvements

- **Better Language Detection**: Currently uses a simple heuristic to detect language; could be improved
- **Instruction Set Detection**: Fixed to 'amd64' currently; should pull from compiler properties
- **Better Error Handling**: Enhance error message display and recovery
- **Additional User Settings**: Allow configuration of explanation preferences

## Current Operation

1. When compilation completes, the Claude Explain tool shows the consent UI
2. User grants consent (stored for the session)
3. Data is sent directly to the Claude Explain API
4. The markdown explanation is displayed in the tool window
5. Subsequent compilations automatically generate new explanations (no need to re-consent)

## Next Steps

1. Test with various languages and compilation scenarios
2. Consider adding persistent consent across browser sessions
3. Evaluate API performance and rate limiting needs
4. Refine instruction set and language detection
5. Address UI styling improvements based on user feedback:
   - Update UI styling for "generating explanation" to work with all themes (especially pink theme)
   - Make "generating explanation" part of the toolbar with proper animation and status colors (similar to `.status-icon` in compiler view)
   - Reduce excessive padding on the bottom bar
6. Add appropriate disclaimers about AI agents and their limitations
7. Fix HTML body not properly updating on page reload (when "#tool-output" shows "The current language or compiler does not support this tool" instead of the explanation content) - appears to be an initialization order issue
8. Implement restriction to allow only one "Claude Explain" tool instance at a time (similar to other unique tools) to prevent duplicate requests
9. Fix issue requiring force-recompilation for HTML to appear in the Claude Explain widget at all (related to tool integration)

## Things to address:
- Sometimes on page reload, the HTML body of the tool doesn't properly update and shows "#tool-output" with "The current language or compiler does not support this tool" instead of the explanation content. This appears to be related to the initialization order.
- Currently allows multiple "Claude Explain" tool instances which can lead to duplicate API requests for the same output. Should be limited to one instance like other unique tools.
- Sometimes requires force-recompilation for HTML content to appear in the Claude Explain widget at all, indicating imperfect integration with the tools framework.
- Better toolbar integration for the "generating explanation" indicator. Just replace with a spinner with no text; that turns red/green on success/failure, like the compiler.ts `.status-icon`
- Proper theming support across all color schemes. we seem to have put the dark theme in the main sccs.
- Disclaimers about AI limitations and potential inaccuracies
