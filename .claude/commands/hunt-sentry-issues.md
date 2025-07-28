# Sentry Issue Hunter

I need you to help me systematically hunt and fix Sentry issues.

**Prerequisites:** Check if you have access to Sentry MCP tools by looking for any tools containing "Sentry" in their name (e.g., `mcp__Sentry__find_issues`, `mcp__Sentry__get_issue_details`). If you cannot find these tools in your available tools list, tell the user to install the Sentry MCP with the following command:
```
claude mcp add-json Sentry '{"command": "npx", "args": ["mcp-remote@latest", "https://mcp.sentry.dev/sse"] }'
```
Sometimes it takes a moment for the Claud code to connect to Sentry, so if the user believes they have already installed the tooling, then ask them to wait, and check with `/mcp` for connection to Sentry before running this custom command.

Think hard and follow this workflow:

## Step 1: Find Issues
Use the Sentry MCP tool `mcp__Sentry__find_issues` to find the top 20 unresolved issues sorted by occurrence count:
- Call it with organizationSlug="compiler-explorer", query="is:unresolved", sortBy="count"
- Present a numbered list of the most impactful issues

## Step 2: Issue Selection  
Let me choose which issue to investigate, or if $ARGUMENTS is provided, investigate that specific issue ID.

## Step 3: Investigation Setup
- Fetch detailed issue information including stack traces using `mcp__Sentry__get_issue_details`
- Create a new branch from origin/main with format: `claude/fix-sentry-ISSUE-ID`
- Use TodoWrite to track progress on this investigation

## Step 4: Root Cause Analysis
- Analyze the stack trace to identify the problematic code location
- Use search tools (Grep, Glob, Read) to examine the relevant files
- Think hard about the root cause - don't just paper over symptoms
- Use the Sentry MCP to gather multiple occurrences and ensure the root cause explains all (or most) instances

## Step 5: Implementation
- Follow CLAUDE.md guidelines for defensive programming and code style
- Ensure errors surface in Sentry rather than failing silently
- Use centralized helper functions rather than spreading type checks

## Step 6: Quality Assurance
- Run `npm run ts-check` to verify TypeScript compliance
- Run `npm run test-min` to ensure no regressions
- Follow pre-commit workflow (never use --no-verify)

## Step 7: Delivery
- Commit with descriptive message referencing the Sentry issue ID
- Push branch and create PR with:
  - Clear title referencing the issue
  - Detailed description explaining root cause and fix
  - Test plan for verification

## Step 8: Iteration
- Mark todo as completed
- Ready to hunt the next issue!

Focus on robust solutions that prevent the error category, not just this specific instance.
