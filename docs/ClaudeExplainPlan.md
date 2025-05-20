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

### 4. Limitations and Future Improvements

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