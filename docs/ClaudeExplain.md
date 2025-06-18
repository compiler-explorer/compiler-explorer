# Claude Explain

Claude Explain is a feature in Compiler Explorer that uses Claude AI to provide natural language explanations of assembly code generation. This feature helps users understand how their source code is translated into assembly, what optimizations are applied, and why certain compiler decisions were made.

## How It Works

1. After compilation, users can click the "Claude Explain" button (robot icon) in the compiler pane toolbar.
2. A dedicated Explain pane opens, prompting users to consent to sending their code and compilation output to the Claude Explain API.
3. Users can customize their explanation by selecting:
   - **Audience Level**: Beginner, Intermediate, or Expert
   - **Explanation Type**: Assembly-focused, Source-to-assembly mapping, or Optimization-focused
4. The system checks for `no-ai` directive in the source code. If found, the explanation feature is disabled with a clear message.
5. Once consent is given (persisted for the browser session), the code, compiler information, and assembly output are sent to the API.
6. Claude analyzes the relationship between the source code and generated assembly, then provides a tailored explanation.
7. The explanation is displayed in markdown format with syntax highlighting in the explain pane.
8. Subsequent compilations automatically update the explanation if the explain pane is open.
9. Responses are cached both client-side and server-side to avoid redundant API calls.

## Configuration

The Claude Explain feature requires minimal configuration:

```ini
# In compiler-explorer.defaults.properties
explainApiEndpoint=https://api.compiler-explorer.com/explain
```

That's it! No tool configuration is needed. The explain button automatically appears in the compiler toolbar when an API endpoint is configured and hides when it's not.

## Privacy Notice

When using Claude Explain:

- Your source code and compilation output are sent to an external API.
- Data is only sent after explicit user consent.
- Consent is remembered for your browser session.
- The API is provided by Anthropic, the makers of Claude.
- Anthropic does not use the data sent to them for training their models.
- Compiler Explorer's privacy policy has been updated to include Claude Explain usage.
- If your source code contains `no-ai` (case-insensitive), it will not be sent to the API, and a special message will be displayed.

## Technical Implementation

The feature consists of:

1. **Server-side configuration**: A single property setting to configure the API endpoint.

2. **Client-side component**:
   - The `ExplainView` class (`static/panes/explain-view.ts`) which extends `Pane` to:
     - Display a consent prompt with clear information about what data is sent
     - Dynamically fetch available audience levels and explanation types from the API
     - Make API requests to the Claude Explain endpoint
     - Cache responses using an LRU cache (200KB limit) to reduce API costs
     - Present the explanation with markdown rendering and syntax highlighting
     - Handle error states and loading indicators
     - Show usage statistics (tokens, cost, model) in a bottom bar
     - Persist user preferences (audience level, explanation type) for the session
   - Uses the `marked` library for markdown rendering
   - Features theme-aware styling (light/dark modes)
   - Includes a reload button to bypass cache and get fresh explanations
   - Provides Bootstrap popovers for option descriptions

3. **Compiler integration**:
   - A button is added to the compiler pane toolbar
   - The button disables when an explain view is already open for that compiler
   - Proper event handling for view lifecycle

4. **Testing**:
   - Simple frontend Cypress tests have been written
   - Tests require the `explainApiEndpoint` to be configured in the test environment
   - Once enabled, tests verify the explain pane opens correctly when the button is clicked

## UI Features

The explain pane provides:

- Clean rendering of markdown content using the `marked` library
- Syntax highlighting for code blocks using Prism.js
- Theme-aware styling that adapts to light/dark modes
- Responsive layout with font scaling support
- Loading states with animated spinner
- Error states with helpful messages
- Customization controls:
  - **Audience Level selector**: Choose between Beginner, Intermediate, or Expert explanations
  - **Explanation Type selector**: Focus on Assembly, Source-to-assembly mapping, or Optimizations
  - **Info buttons**: Click the info icon next to each selector to see descriptions of each option
- Bottom status bar showing:
  - AI model used (e.g., claude-3-opus-20240229)
  - Token usage (input/output/total)
  - Estimated cost (input/output/total)
  - Cache status indicator (client cache, server cache, or fresh generation)
  - Reload button to bypass all caches
- Session-persistent consent (you only need to consent once per browser session)

## API Integration

### GET Request - Available Options

On first load, the client fetches available options:

```
GET /
```

Returns:
```json
{
  "audience": [
    {
      "value": "beginner",
      "description": "For beginners learning assembly language. Uses simple language and explains technical terms."
    },
    {
      "value": "intermediate",
      "description": "For users familiar with basic assembly concepts. Focuses on compiler behavior and choices."
    },
    {
      "value": "expert",
      "description": "For advanced users. Uses technical terminology and covers advanced optimizations."
    }
  ],
  "explanation": [
    {
      "value": "assembly",
      "description": "Explains the assembly instructions and their purpose."
    },
    {
      "value": "source",
      "description": "Explains how source code constructs map to assembly instructions."
    },
    {
      "value": "optimization",
      "description": "Explains compiler optimizations and transformations applied to the code."
    }
  ]
}
```

### POST Request - Generate Explanation

The client sends a POST request to generate an explanation:

```json
{
  "language": "c++",
  "compiler": "GCC 13.2",
  "code": "Source code",
  "compilationOptions": ["-O2", "-std=c++20"],
  "instructionSet": "amd64",
  "asm": ["Assembly output lines"],
  "audience": "intermediate",        // Optional, default: "beginner"
  "explanation": "optimization",     // Optional, default: "assembly"
  "bypassCache": false               // Optional, default: false
}
```

The API returns:

```json
{
  "status": "ok",
  "explanation": "Markdown-formatted explanation",
  "model": "claude-3-opus-20240229",
  "usage": {
    "inputTokens": 500,
    "outputTokens": 300,
    "totalTokens": 800
  },
  "cost": {
    "inputCost": 0.0015,
    "outputCost": 0.0045,
    "totalCost": 0.006
  },
  "cached": false  // Indicates if served from server-side cache
}
```

## Caching

The explain feature uses multi-level caching to reduce API costs and improve response times:

### Client-Side Cache
- Responses are cached using an LRU (Least Recently Used) cache
- Cache key is based on the hash of the API request payload (including audience and explanation type)
- Cache has a 200KB size limit
- Stored in browser memory for the session

### Server-Side Cache
- The API server also caches responses
- Cached responses are indicated by `cached: true` in the API response
- Server cache is shared across all users

### Cache Bypass
- Users can bypass all caches using the reload button in the bottom bar
- This sends `bypassCache: true` to the API, forcing fresh generation
- Fresh responses are still written to both caches for future use

### Cache Status Display
- The bottom bar shows cache status:
  - 🔄 "Cached (client)" - Served from browser cache
  - 🔄 "Cached (server)" - Served from API server cache
  - ✨ "Fresh" - Newly generated explanation

## Limitations

- Claude may not be able to explain every compiler optimization or assembly pattern.
- Large assemblies may be truncated before being sent to the API.
- The feature requires an internet connection to access the external API.
