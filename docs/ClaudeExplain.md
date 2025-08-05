# Claude Explain

Claude Explain uses Claude AI to provide natural language explanations of assembly code generation, helping users understand how their source code is translated into assembly and what compiler optimizations are applied.

## How It Works

1. Click the "Explain" button in the compiler toolbar to open a dedicated explanation pane
2. After compilation completes:
   - Failed compilations show "Cannot explain: Compilation failed"
   - Code with `no-ai` directive shows a special message
   - Otherwise, users consent to sending code and compilation output to the Claude API
3. Customize explanations by selecting audience level (beginner/intermediate/expert) and explanation type (assembly/source/optimization)
4. Once consent is given (persisted for the session), Claude analyzes the code and assembly relationship
5. The markdown-formatted explanation appears with syntax highlighting
6. Responses are cached client-side (LRU, 200KB limit) and server-side to reduce API costs
7. Use the reload button to bypass caches and get fresh explanations

## Configuration

Add the API endpoint URL to `compiler-explorer.*.properties`:

```ini
explainApiEndpoint=https://api.compiler-explorer.com/explain
```

The explain button appears automatically when configured.

## Privacy Notice

- Source code and compilation output are sent to Anthropic's Claude API after explicit user consent
- Consent is remembered for the browser session (not stored in cookies/localStorage)
- Anthropic does not use the data for model training
- Code containing `no-ai` (case-insensitive) is never sent to the API
- Compiler Explorer's privacy policy covers Claude Explain usage

## Technical Implementation

**Server-side**: Single property configuration (API endpoint). Server code: https://github.com/compiler-explorer/explain

**Client-side**:
- `ExplainView` class (`static/panes/explain-view.ts`) handles UI, consent, API requests, and caching
- `explain-view-utils.ts` contains testable business logic (validation, formatting, request building)
- Uses `marked` library for markdown rendering with syntax highlighting
- LRU cache (200KB limit) shared across all explain views in the session
- Theme-aware styling with responsive layout and font scaling

**Features**:
- Loading states with animated spinner
- Error handling with helpful messages
- Audience/explanation type selectors with Bootstrap popovers
- Status bar showing model, token usage, cost estimates, and cache status
- Session-persistent consent and user preferences
- Reload button to bypass all caches

**Testing**:
- Comprehensive Cypress E2E tests covering UI interactions, consent flow, API mocking, caching behavior, and error handling
- Tests verify explain pane functionality, theme persistence, and proper handling of compilation states

## API Integration

**GET /** - Fetch available options:
```json
{
  "audience": [
    {"value": "beginner", "description": "Simple language, explains technical terms"},
    {"value": "intermediate", "description": "Focuses on compiler behavior and choices"},
    {"value": "expert", "description": "Technical terminology, advanced optimizations"}
  ],
  "explanation": [
    {"value": "assembly", "description": "Explains assembly instructions and purpose"},
    {"value": "source", "description": "Maps source code constructs to assembly"},
    {"value": "optimization", "description": "Explains compiler optimizations and transformations"}
  ]
}
```

**POST /** - Generate explanation:
```json
{
  "language": "c++",
  "compiler": "GCC 13.2",
  "code": "Source code",
  "compilationOptions": ["-O2", "-std=c++20"],
  "instructionSet": "amd64",
  "asm": ["Assembly output lines"],
  "audience": "intermediate",
  "explanation": "optimization",
  "bypassCache": false
}
```

Optional fields: `audience` (default: "beginner"), `explanation` (default: "assembly"), `bypassCache` (default: false)

Response:
```json
{
  "status": "success",
  "explanation": "Markdown-formatted explanation",
  "model": "claude-3-sonnet",
  "usage": {"inputTokens": 500, "outputTokens": 300, "totalTokens": 800},
  "cost": {"inputCost": 0.0015, "outputCost": 0.0045, "totalCost": 0.006},
  "cached": false
}
```

## Caching

**Multi-level caching** reduces API costs and improves response times:

- **Client-side**: LRU cache, cache key from request payload hash
- **Server-side**: Shared cache across users, indicated by `cached: true` in response
- **Cache bypass**: Reload button sends `bypassCache: true` for fresh generation
- **Status display**: Shows cache state, models, token usage and cost estimates

## Limitations

- May not explain every compiler optimization or assembly pattern
- Large assemblies may be truncated before sending to API
- Requires internet connection for external API access
- One explain view per compiler at a time