# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands
- Build: `npm run webpack`, `npm start`
- Dev Mode: `make dev`, `make gpu-dev`
- Lint: `npm run lint` (auto-fix), `npm run lint-check` (check only)
- Type Check: `npm run ts-check`
- Test: `npm run test` (all), `npm run test-min` (minimal)
- Test Single File: `npm run test -- --run base-compiler-tests.ts`
- Test Specific Pattern: `npm run test -- -t "should handle execution failures"`
- Cypress Tests: `npm run cypress`
- Pre-commit Check: `make pre-commit` or `npm run check`

## Important Workflow Requirements
- ⚠️ NEVER BYPASS PRE-COMMIT HOOKS! NEVER use `git commit -n` or `--no-verify` ⚠️
- ALWAYS run `make pre-commit` or at minimum `npm run ts-check` and `npm run lint` before committing
- The full process must always be:
  1. Make changes
  2. Run `npm run ts-check` to verify TypeScript types
  3. Run `npm run lint` to fix style issues (will auto-fix many problems)
  4. Run `npm run test` to verify functionality (or at least `npm run test-min`)
  5. ONLY THEN commit changes with plain `git commit` (NO FLAGS!)
- Bypassing these checks will lead to broken builds, failed tests, and PRs that cannot be merged
- ALWAYS use HEREDOC syntax for complex shell commands, especially those containing quotes, newlines, or special characters:
  ```bash
  gh pr create --title "Title" --body "$(cat <<'EOF'
  PR body content with "quotes" and special chars like $variables
  that would otherwise need escaping
  EOF
  )"
  ```

## Style Guidelines
- TypeScript: Strict typing, no implicit any, no unused locals
- Formatting: 4-space indentation, 120 char line width, single quotes
- No semicolon omission, prefer const/let over var
- Client-side: TypeScript transpiled to ES5 JavaScript. This process requires import of `blah.js` even though `blah.ts` is the actual filename
- ALWAYS place imports at the top of files, never inside functions or methods, unless absolutely necessary (and confirm before proposing)
- Use Underscore.js for utility functions
- Write tests for new server-side components
- Where appropriate suggest follow-up improvements to code to improve code quality, and DRY up where feasible
- Documentation is in `docs/` directory; update where necessary, in particular if anything about the RESTful API changes
- Don't add comments above code that's clearly self-documenting. For example, don't put comments like this:
  ```
  // Initialises the thing
  initialiseThing();
  ```
- Avoid redundant function header comments that merely repeat the function name. For example:
  ```
  /**
   * Sets up compiler change handling
   */
  function setupCompilerChangeHandling() {...}
  ```
  In this case, the function name already clearly states what it does.
- Comments should provide additional context or explain "why" something is done, not just restate "what" is being done.
- Only add function header comments when they provide meaningful information beyond what the function name and signature convey.
- Use British English spellings for things like "initialise" and "colour", but only in new code. It's a preference not a hard requirement
- Use modern Typescript features like optional chaining when updating existing code or adding new code

## Testing Guidelines
- Use Vitest for unit tests (compatible with Jest syntax)
- Tests are in the `/test` directory, typically named like the source files they test
- Use spy functions with `vi.spyOn()` for mocking dependencies
- Test structure follows describe/it pattern with descriptive test names
- Separate tests with clear section headers using comments for readability
- Consider cross-platform compatibility (especially Windows path handling)
- For complex files, organize tests by functionality rather than by method
- Use `beforeEach`/`afterEach` to set up and clean up test environment
- Remember to restore mocks with `vi.restoreAllMocks()` after tests
- Test both success and error cases
- Coverage is available with `npm test:coverage`
- For Windows-specific path issues, either:
  - Skip tests with `if (process.platform === 'win32') return;`
  - Write platform-specific assertions
  - Use path-agnostic checks

## Compiler Testing Specifics
- Mock filesystem operations when testing file I/O
- Use `makeFakeCompilerInfo()` for creating test compiler configurations
- Use `makeCompilationEnvironment()` to create test environments
- Mock `exec` calls for testing compilation and execution
- For BaseCompiler, use the test utils from `/test/utils.js`
- Test specific combinations of compiler capabilities
- Focus tests on behavior, not implementation details
- Use platform-agnostic assertions where possible