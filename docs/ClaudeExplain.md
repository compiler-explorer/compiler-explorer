# Claude Explain

Claude Explain is a feature in Compiler Explorer that uses Claude AI to provide natural language explanations of assembly code generation. This feature helps users understand how their source code is translated into assembly, what optimizations are applied, and why certain compiler decisions were made.

## How It Works

1. After compilation, users can select the "Claude Explain" tool from the tool dropdown.
2. Upon selection, users are prompted to consent to sending their code and compilation output to the Claude Explain API.
3. Once consent is given, the code, compiler information, and assembly output are sent to the API.
4. Claude analyzes the relationship between the source code and generated assembly, then provides a detailed explanation.
5. The explanation is displayed in markdown format in the tool pane.

## Configuration

The Claude Explain feature is configured in the properties files:

```ini
# In compiler-explorer.defaults.properties
explainApiEndpoint=https://api.compiler-explorer.com/explain

# In language-specific property files (e.g., c++.defaults.properties)
tools=<other tools>:explain

tools.explain.name=Claude Explain
tools.explain.type=postcompilation
tools.explain.stdinHint=disabled
tools.explain.languageId=markdown
```

## Privacy Notice

When using Claude Explain:

- Your source code and compilation output are sent to an external API.
- Data is only sent after explicit user consent.
- The API is provided by Anthropic, the makers of Claude.
- The API may store information to improve its service.

## Technical Implementation

The feature consists of:

1. **Server-side configuration**: Minimal property settings to enable the tool and configure the API endpoint.

2. **Client-side component**: The `ExplainView` class which extends the standard `Tool` class to:
   - Display a consent prompt
   - Make API requests to the Claude Explain endpoint
   - Present the explanation with proper markdown formatting
   - Handle error states and loading indicators

3. **Hub integration**: Special handling in the hub factory method to use the `ExplainView` for explain tools.


## Limitations

- Claude may not be able to explain every compiler optimization or assembly pattern.
- Large assemblies may be truncated before being sent to the API.
- The feature requires an internet connection to access the external API.
- Currently available for C++ and other major languages.