# GitHub Issue Hunter

I need you to help me systematically hunt and fix bugs reported in GitHub.

When accessing GitHub, you should use the MCP tooling named `github` for preference as this has
the most convenient API for finding issues.

Think hard and follow this workflow:

## Step 1: Find Bugs
Search GitHub using the MCP `mcp__github__search_issues` command. Due to token limits, use a multi-step approach:

1. Make multiple calls with small page sizes:
   - q="repo:compiler-explorer/compiler-explorer is:issue state:open type:bug sort:created-asc"
   - perPage=2 (very small to avoid token limits)
   - page=1, then page=2, etc.

2. For each page of results, use the Task tool to:
   - Extract just the essential info (issue number, title, created date)
   - Summarize each issue in 1-2 sentences
   - Return a concise list

3. Repeat until you have collected ~12 bug summaries total (e.g., 6 calls × 2 issues each)

4. Present a numbered list of all bugs found, focusing on impact and age

Note: The search uses `type:bug` which is GitHub's issue type filter (different from labels).

## Step 2: Issue Selection  
Let me choose which issue to investigate.

## Step 3: Investigation Setup
- Fetch detailed issue information on the bug, comments, labels and so on.
- Ask the user for clarifying information if it's not clear.
- Create a new branch from origin/main with format: `claude/fix-ISSUE-ID`
- Use TodoWrite to track progress on this investigation

## Step 4: Root Cause Analysis
- Use search tools (Grep, Glob, Read) to examine the relevant files
- Think hard about the root cause - don't just paper over symptoms

## Step 5: Implementation
- Follow CLAUDE.md guidelines for defensive programming and code style
- Write tests where possible; don't overcomplicate code. Ask the user if unsure
- If the issue relates to UI, ask the user for help with testing manually
- Do not assume your changes fix the issue until positively confirmed by the user

## Step 6: Delivery
- Commit with descriptive message referencing the Sentry issue ID
- Push branch and create PR with:
  - Clear title referencing the issue
  - Detailed description explaining root cause and fix
  - Test plan for verification

## Step 7: Iteration
- Mark todo as completed
- Ready to hunt the next issue!

Focus on robust solutions that prevent the error category, not just this specific instance.
