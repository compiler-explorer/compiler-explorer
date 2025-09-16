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

## Architecture Guidelines
- **Frontend/Backend Separation**: Frontend code (`static/`) MUST NOT import from backend code (`lib/`)
  - Frontend should use API calls to communicate with backend
  - Shared types should be imported from `types/` directory instead
  - This separation is enforced by pre-commit hooks (`npm run check-frontend-imports`)
  - Violations will cause build failures and prevent commits

## Worker Mode Configuration
- **Compilation Workers**: New feature for offloading compilation tasks to dedicated worker instances
  - `compilequeue.is_worker=true`: Enables compilation worker mode (similar to execution workers)
  - `compilequeue.queue_url`: SQS queue URL for compilation requests (both regular and CMake)
  - `compilequeue.events_url`: WebSocket URL for sending compilation results
  - `compilequeue.worker_threads=2`: Number of concurrent worker threads
  - `compilequeue.poll_interval_ms=1000`: Interval between poll attempts after processing or errors (default: 1000ms). Note: SQS long polling means actual wait time is up to 20 seconds when queue is empty
  - `--instance-color <color>`: Optional command-line parameter to differentiate deployment instances. When specified (blue or green), modifies the queue URL by appending the color to the queue name (e.g., `staging-compilation-queue-blue.fifo`)
- **Implementation**: Located in `/lib/compilation/sqs-compilation-queue.ts` with shared parsing utilities in `/lib/compilation/compilation-request-parser.ts`
- **Queue Architecture**: Uses single AWS SQS FIFO queue for reliable message delivery, messages contain isCMake flag to distinguish compilation types
- **S3 Overflow Support**: Large compilation requests exceeding SQS message size limits (256KB) are automatically stored in S3
  - Messages exceeding the limit are stored in S3 bucket `compiler-explorer-sqs-overflow`
  - SQS receives a lightweight reference message with type `s3-overflow` containing S3 location
  - Workers automatically detect overflow messages and fetch the full request from S3
  - S3 objects are automatically deleted after 1 day via lifecycle policy
- **Result Delivery**: Uses WebSocket-based communication via `PersistentEventsSender` for improved performance with persistent connections
- **Message Production**: Queue messages are produced by external Lambda functions, not by the main Compiler Explorer server
- **Shared Parsing**: Common request parsing logic is shared between web handlers and SQS workers for consistency
- **Remote Compiler Support**: Workers automatically detect and proxy requests to remote compilers using HTTP, maintaining compatibility with existing remote compiler infrastructure
- **S3 Storage Integration**: Compilation results include an `s3Key` property containing the cache key hash for S3 storage reference. Large results (>31KiB) can be stored in S3 and referenced by this key. The s3Key is removed from API responses before sending to users.
- **Metrics & Statistics**: SQS workers track separate Prometheus metrics (`ce_sqs_compilations_total`, `ce_sqs_executions_total`, `ce_sqs_cmake_compilations_total`, `ce_sqs_cmake_executions_total`) and record compilation statistics via `statsNoter.noteCompilation` for Grafana monitoring, mirroring the regular API route behavior.

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

### Test Execution with Expensive Test Skipping
- The `SKIP_EXPENSIVE_TESTS=true` environment variable skips expensive tests (like filter tests)
- Pre-commit hooks use `vitest related` to run only tests related to changed files
- Use `npm run test-min` to run tests with expensive tests skipped
- Use `npm run test` to run all tests including expensive ones
- To mark tests as expensive, use: `describe.skipIf(process.env.SKIP_EXPENSIVE_TESTS === 'true')('Test suite', () => {...})`

## Compiler Testing Specifics
- Mock filesystem operations when testing file I/O
- Use `makeFakeCompilerInfo()` for creating test compiler configurations
- Use `makeCompilationEnvironment()` to create test environments
- Mock `exec` calls for testing compilation and execution
- For BaseCompiler, use the test utils from `/test/utils.js`
- Test specific combinations of compiler capabilities
- Focus tests on behavior, not implementation details
- Use platform-agnostic assertions where possible
