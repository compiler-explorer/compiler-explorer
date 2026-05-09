# MCP endpoint

Compiler Explorer ships a built-in [Model Context Protocol](https://modelcontextprotocol.io)
server that lets LLM clients drive CE the same way the web UI does: list languages and
compilers, compile (and optionally execute) source, look up assembly instruction docs,
and round-trip short URLs.

The endpoint is unauthenticated and lives alongside the public REST API on every CE
deployment.

## Connection details

- URL: `https://godbolt.org/mcp` (or `<your-ce-host>/mcp` for self-hosted).
- Transport: [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http),
  stateless (one JSON-RPC POST per call, no session id).
- Methods accepted: `POST` for tool calls; `OPTIONS` for CORS preflight. `GET` and
  `DELETE` return `405 Method Not Allowed`.
- Auth: none.

## Using it from a Claude client

In Claude Desktop or Claude Code, add a custom HTTP MCP server pointing at
`https://godbolt.org/mcp`. No credentials required.

For Claude Code on the command line:

```sh
claude mcp add --transport http compiler-explorer https://godbolt.org/mcp
```

## Tools

All tools return JSON in a single `text` content block. Errors set `isError: true` and
put a human-readable message in the same block.

### `list_languages`

Returns every supported language, with its default compiler id and the number of
compilers that target it. Read-only.

### `list_compilers`

List compilers, optionally filtered by `language`, `instructionSet`, or a free-text
`match`. The `latestPerMajor` flag reduces the firehose to "newest stable per
(language, arch, semver major), all nightly + prerelease, no experimentals" — the
right default when an LLM is picking a compiler. `lean: true` drops everything but
id and name. Read-only.

### `list_libraries`

List libraries available for a given language, with the same `match`/`maxResults`/`lean`
controls as `list_compilers`. Read-only.

### `compile`

Compile a source string with a chosen `compiler` and `options`, optionally linking
`libraries` and optionally executing the result. With `execute: true` the program
runs in CE's ephemeral sandbox; runtime output appears at the top level and the
compile diagnostics move to `buildResult`. Output is line-capped (defaults: 500 asm,
100 stdout, 100 stderr) — raise the `maxAsmLines` / `maxStdoutLines` / `maxStderrLines`
caps to retrieve more. Read-only from the connector's point of view; sandbox effects
do not escape the call.

### `lookup_asm_instruction`

Retrieve documentation for an assembly mnemonic in a given instruction set
(`amd64`, `arm64`, `riscv`, …). Read-only.

### `generate_short_url`

Persist a Compiler Explorer short URL that captures source, language, compiler, options,
and libraries, and return its `https://godbolt.org/z/...` form. Additive only — repeat
calls with the same payload return the same URL (the storage layer dedupes by config
hash). Marked as a write tool with `idempotentHint: true`.

### `get_shortlink_info`

Resolve an `https://godbolt.org/z/<id>` URL (or just the id) back into source plus
compiler config. The returned compiler entries use the same shape `compile` accepts
(`{compiler, options, libraries:[{id, version}]}`), so a shortlink can be re-compiled
without translation. Multi-pane shortlinks (executors, conformance views, CMake trees)
are flattened to the basic compile inputs. Read-only.

## Notes for tool authors

- All tools carry MCP `annotations` (`title`, `readOnlyHint`/`destructiveHint`,
  `openWorldHint: false`); see `lib/mcp/tools/*.ts`.
- The server uses `StreamableHTTPServerTransport` in stateless mode
  (`sessionIdGenerator: undefined`). Each POST is a complete request/response cycle.
- Tool implementations live in `lib/mcp/tools/` and are wired up in
  `lib/mcp/index.ts`. Tests live in `test/mcp/`.

## Reporting issues

File MCP-specific bugs at <https://github.com/compiler-explorer/compiler-explorer/issues>.
