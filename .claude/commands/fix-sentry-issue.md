# Fix Specific Sentry Issue

I need you to investigate and fix Sentry issue: $ARGUMENTS

**Prerequisites:** Check if you have access to Sentry MCP tools by looking for any tools containing "Sentry" in their name (e.g., `mcp__Sentry__find_issues`, `mcp__Sentry__get_issue_details`). If you cannot find these tools in your available tools list, tell the user to install the Sentry MCP with the following command:
```
claude mcp add --scope local Sentry --transport sse https://mcp.sentry.dev/sse
```
Sometimes it takes a moment for the Claud code to connect to Sentry, so if the user believes they have already installed the tooling, then ask them to wait, and check with `/mcp` for connection to Sentry before running this custom command.

Think hard and follow this systematic approach:

## Investigation
1. Use `mcp__Sentry__get_issue_details` to fetch full details for issue $ARGUMENTS using organization slug `compiler-explorer`
2. Create branch: `claude/fix-sentry-$ARGUMENTS` from origin/main
3. Analyze stack trace to identify root cause
4. Use TodoWrite to track this specific fix

## Implementation
1. Search codebase to locate problematic code
2. Think about root cause - don't paper over bugs
3. Follow CLAUDE.md guidelines for defensive programming, and code style

## Validation
1. Run `npm run ts-check` and `npm run test-min`
2. Commit with message: "Fix [description] - Fixes $ARGUMENTS"
3. Push and create PR with detailed explanation, referencing the Sentry issue.

Focus on robust solutions that prevent the error category, not just this specific instance.
Use the Sentry MCP to gather multiple occurrences of the issue and ensure the root cause
explains all (or most) of the instances seen, and has no counterfactuals.
