# Gooo activity to Go source

This compiler entry displays the generated Go function for one selected Gooo activity body. It does not compile a whole program, produce target assembly, or execute generated Go. Binary output and execution are disabled, including executor API requests.

The editor contains a `.gooo` declaration file. The visible compiler options select the activity: `--activity Double`. This entry accepts exactly `--activity <identifier>`; other flags and extra arguments are refused. The default example uses the tested single argument name `input`. Edit its `computes` body to `return input - 1` to inspect a different emitted function. Syntax highlighting is not a claim that every highlighted construct is supported by body code generation.

The initial entry pins Gooo `0.6.32-dev` at compiler source `83928fa4538245abe33b08549da47f8cbdc50ed2`. It invokes `body-codegen --json --activity <name> <input file>`, validates the v3 report and actual input/output SHA256 digests, and displays only the validated `source` string as Go. The report must state successful typechecking, deterministic emission replay, the pinned producer and zero repository writes. These checks bind compiler output consistently; they are not independent authentication or proof of generated-code execution.

A flat `FAIL_CLOSED` report, invalid report, timeout, subprocess error or digest mismatch produces a diagnostic with no generated source. No instruction addresses or Gooo-to-Go source mappings are fabricated. The checked examples cover selected scalar activity bodies; other body profiles remain unverified. Model-routing URL and API-key settings are blanked for these compiler subprocesses while CE sandbox settings are preserved.

The saved regression fixtures are real darwin/arm64 output from the pinned CLI, including valid, repeated, changed-body and undefined-`input0` refusal cases. A local CE HTTP API smoke with that actual CLI also checked compiler discovery, identical repeated output, changed-body output, invalid options and explicit executor refusal. An unsupported `execute` filter is ignored and returns source without executing it; this differs from an explicit executor request, which returns a diagnostic.

A [verified Linux amd64 smoke run](https://github.com/kimjooyoon/compiler-explorer/actions/runs/39154930655) checked the pinned 839 compiler through five direct CLI invocations and eight real CE `app.ts` HTTP requests. The CLI checks covered build identity, valid/repeated/changed emission and flat refusal. The HTTP checks covered discovery, valid/repeated/changed source, invalid input/options, explicit executor refusal and an ignored execution filter. This verifies the selected scalar activity body and its typecheck/emission behaviour; generated Go was not executed. Production nsjail operation and hosted availability remain unverified. Installer packaging and complete dependency notices remain follow-up requirements.

## Use the existing MCP endpoint

On a self-hosted Compiler Explorer instance with this backend and its pinned compiler configured, use the existing [`/mcp` endpoint](MCP.md). Discover the Gooo compiler with `list_compilers` and `language: "gooo"`, then inspect its `supportsExecute` and `supportsBinary` values. The compiler has the `experimental` release track; when using `latestPerMajor: true`, also set `includeExperimental: true` to include it.

Call `compile` with `language: "gooo"`, the returned compiler ID, `options: "--activity Double"`, `execute: false`, and the source from `examples/gooo/default.gooo`. The tool returns JSON inside its text content. For this backend, `asm` contains Go function source. Check the embedded `code`: a successful JSON-RPC response or an absent `isError` does not mean that compilation passed. A nonzero compilation code is a refusal; preserve its diagnostic.

This uses the same source-output backend through Compiler Explorer's existing MCP transport. It does not execute generated Go or establish that Gooo is installed on a public Compiler Explorer deployment.

Compiler source: <https://github.com/kimjooyoon/meta-ontology-go/tree/83928fa4538245abe33b08549da47f8cbdc50ed2> (MIT). The existing language request is <https://github.com/compiler-explorer/compiler-explorer/issues/9258>.
