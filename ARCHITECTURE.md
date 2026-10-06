# ARCHITECTURE.md

A map of the Compiler Explorer (CE) codebase for AI agents and new contributors. It answers "where does X live",
"how does a request flow", and "what do I touch to add Y". It deliberately avoids line numbers: grep for the symbol
names given here, since the big files drift. Workflow rules (pre-commit, lint, commit etiquette) are in
[AGENTS.md](AGENTS.md); task-specific how-tos are in [docs/](docs/). This file sits between them.

## 1. Big picture

CE is a TypeScript/Node.js web app. The browser sends source code plus options to an Express server, the server runs a
real compiler binary in a sandbox, parses its output (assembly, IR, diagnostics, binaries), and returns JSON that the
browser renders in draggable panes. There are no user accounts; all UI state is serialised into the URL or a shortlink.

```
 Browser (static/)                                                  Server (app.ts, lib/)
 +--------------------------------+  POST /api/compiler/:id/compile  +----------------------------------------------+
 | Editor pane --editorChange-->  | -------------------------------> | CompileHandler.handle                        |
 | Compiler pane                  |                                  |   -> compilerFor() -> parseRequest()         |
 |   compile() -> CompilerService |                                  |   -> BaseCompiler.compile()                  |
 | <--compileResult-- (EventHub)  | <------------- JSON ------------ |      cache? -> enqueue -> doCompilation      |
 | IR/Opt/AST/... views listen    |                                  |      -> exec (nsjail) -> processAsm          |
 +--------------------------------+                                  +----------------------------------------------+
        GoldenLayout + Monaco                                          etc/config/*.properties drive everything
```

Two deployment shapes share this code: godbolt.org (AWS, `--env amazon`, thousands of compilers under
`/opt/compiler-explorer`, installed by the sibling [infra](https://github.com/compiler-explorer/infra) repo) and
local/private installs (`--env dev` by default, compilers from `*.local.properties`). Keep both working.

## 2. Repository map

| Path | What it is |
|---|---|
| `app.ts` | Server entry point. Parses CLI, loads config, calls `initialiseApplication`. |
| `compiler-args-app.ts` | Standalone CLI to debug compiler argument parsers (`docs/Compiler-Args-Debugging.md`). |
| `lib/` | All server-side code. See section 3. |
| `lib/app/` | Startup wiring: CLI, config hierarchy, Express setup, routes, discovery, rescans. |
| `lib/handlers/` | HTTP handlers: `compile.ts` (the compile endpoint), `api.ts`, `route-api.ts` (shortlink pages), `api/*` controllers. |
| `lib/compilers/` | ~165 `BaseCompiler` subclasses, one per compiler family. Registered via `_all.ts`. |
| `lib/parsers/` | TypeScript assembly parsers (`asm-parser.ts` and its subclasses) and pass-dump parsers. Production routes the mainstream languages to the C++ parser instead; see section 3.7. |
| `lib/{demangler,objdumper,formatters,tooling,storage,shortener,asm-docs,buildenvsetup,external-parsers,execution,cfg,runtime-tools}/` | Pluggable families, each with an `_all.ts` registry. See section 3.6. |
| `lib/compilation/` | Shared request parsing and the SQS compilation-worker mode. |
| `lib/cache/` | Compilation/executable caches: InMemory, OnDisk, S3, Multi, Null. |
| `static/` | All browser code. `main.ts` boots, `hub.ts` owns panes, `panes/` are the UI components. See section 5. |
| `static/panes/` | One file per pane type (`compiler.ts`, `editor.ts`, `*-view.ts`, ...) plus `pane.ts` base classes. |
| `static/widgets/` | Reusable UI bits inside panes (compiler picker, libs, overrides, font scale, ...). |
| `static/modes/` | Monaco syntax definitions for ~60 languages. |
| `views/` | Pug templates rendered server-side. `templates/` holds the hidden pane HTML the frontend clones. |
| `types/` | Types shared by frontend and backend (the only `lib`-adjacent code `static/` may import). |
| `shared/` | Runtime code shared by both sides: URL/state serialisation, rison, build-system descriptors. |
| `etc/config/` | ~225 `.properties` files. The configuration system. See section 4. |
| `etc/{nsjail,firejail,cewrapper}/` | Sandbox profiles selected by `execution.*.properties`. nsjail is what Linux production uses; firejail is legacy, cewrapper is Windows-only. |
| `etc/scripts/` | Dev tooling: `ce-properties-wizard`, `check-frontend-imports.js`, `check-license-headers.js`, docenizers, `find-node`. |
| `examples/<lang>/` | Default source per language; `default.<ext>` is the editor's starting code. |
| `test/` | Vitest unit tests (backend). `static/tests/` holds frontend unit tests; `cypress/` holds E2E. See section 7. |
| `docs/` | Human-facing how-tos. `docs/internal/` is for maintainers. |
| `out/` | Build output (gitignored): `out/dist` (tsc), `out/webpack/static` (bundles), `out/compiler-cache`. |

Approximate sizes: `lib/compilers` ~24k lines, `static/panes` ~19k, `lib/base-compiler.ts` alone ~4.4k,
`static/panes/compiler.ts` ~3.8k. Those two files are where most compile-path behaviour lives.

## 3. Backend

### 3.1 Startup sequence

`app.ts` -> `lib/app/cli.ts` `parseArgsToAppArguments` -> `lib/app/config.ts` `loadConfiguration` ->
`lib/app/main.ts` `initialiseApplication`, which in order:

1. Sets up the temp dir, AWS/SSM config, Sentry, wine, remote-execution architectures.
2. Builds the `CompilationEnvironment`, `CompilationQueue`, `FormattingService` and `CompileHandler`
   (`lib/app/compilation-env.ts`).
3. Creates `ClientOptionsHandler` (what the browser gets), the storage backend (`storageSolution`), `CompilerFinder`.
4. Registers controllers and handlers (`lib/app/controllers.ts`).
5. Discovers compilers (`lib/app/compiler-discovery.ts`): either `--prediscovered <json>` (skip version probing, no
   rescans) or `CompilerFinder.find()`. `--discoveryonly <json>` dumps the result and exits.
6. Builds the Express app (`lib/app/server.ts` `setupWebServer`, `lib/app/server-config.ts`) and mounts routes
   (`lib/app/routes-setup.ts`).
7. Starts periodic compiler rescans (`lib/app/compiler-changes.ts`, `rescanCompilerSecs`), the optional Prometheus
   `--metricsPort` server, and SQS worker threads if `execqueue.is_worker` / `compilequeue.is_worker`.
8. Listens (`lib/app/server-listening.ts`; systemd socket activation is supported).

Useful CLI flags (all in `cli.ts`): `--env <envs...>` (default `dev`), `--language <ids...>`, `--port`, `--no-local`,
`--no-cache`, `--no-remote-fetch`, `--debug`, `--prop-debug` (log every property override), `--dev-mode`
(default when `NODE_ENV !== 'production'`; serves the frontend through webpack-dev-middleware).

### 3.2 HTTP routes

Pages and state (`lib/app/server-config.ts`, `lib/handlers/route-api.ts`):

| Route | Purpose |
|---|---|
| `GET /`, `/e`, `/embed-ro`, `/noscript/*` | Rendered pug pages (`lib/app/rendering.ts`). Options are injected into a `div#config`. |
| `GET /z/:id` | Stored shortlink: storage -> `configToGoldenLayout` -> render with `config` pre-injected. |
| `GET /z/:id/code/:session` | Raw source from a shortlink. |
| `GET /clientstate/<base64>` | Full state inline (API-style `ClientState`, normalised by `lib/clientstate-normalizer.ts`). |
| `GET /resetlayout/:id`, `/fromsimplelayout`, `/g/:id` | Layout repair, simple query-string layout, legacy goo.gl redirect. |

API (`lib/handlers/api.ts` `ApiHandler`, controllers in `lib/handlers/api/`; full spec in `docs/API.md`):

| Route | Handler |
|---|---|
| `GET /api/compilers[/:lang]`, `/api/languages`, `/api/libraries[/:lang]`, `/api/tools/:lang` | `ApiHandler` (cached, `apiMaxAgeSecs`) |
| `POST /api/compiler/:id/compile` | `CompileHandler.handle` |
| `POST /api/compiler/:id/cmake`, `/build/:buildSystem` | `CompileHandler.handleCmake` / `handleBuildProject` |
| `POST /api/shortener`, `GET /api/shortlinkinfo/:id` | storage handler via `urlShortenService` |
| `GET /api/asm/:arch/:opcode` | `assembly-documentation-controller.ts` |
| `POST /api/format/:tool`, `GET /api/formats` | `formatting-controller.ts` |
| `GET /api/popularArguments/:id`, `/api/version`, `/api/releaseBuild`, `/api/siteTemplates` | misc |
| `GET /source/:source/list`, `/source/:source/load/:lang/:file` | `source-controller.ts` (the Examples tab) |
| `GET /healthcheck` | `healthcheck-controller.ts`; mounted before logging middleware |
| `POST /mcp` | `lib/mcp/` Model Context Protocol server (`docs/MCP.md`) |

Body parsing: `express.json` is global; the compile route adds a catch-all `express.text`, so `req.body` may be an
object (JSON API) or a string (raw source with options in the query string). `parseRequestReusable` in
`lib/handlers/compile.ts` handles both and is shared with the SQS worker.

### 3.3 The compile pipeline (`lib/base-compiler.ts`)

`CompileHandler.handle` -> `compilerFor()` (id, then alias; remote compilers with `remote` set are proxied over HTTP,
not compiled) -> `parseRequest()` -> `BaseCompiler.compile(source, options, backendOptions, filters, bypassCache,
tools, executeParameters, libraries, files)`.

Inside `BaseCompiler`:

```
compile()
 |- checkOptions / fixFiltersBeforeCacheKey / getCacheKey
 |- env.cacheGet  -- hit -->  (optionally handleExecution)  -->  return
 '- env.enqueue(job)            # CompilationQueue, abandonIfStale
      |- preProcess, newTempDir, writeAllFiles
      |- doCompilation()
      |    |- getOutputFilename, setupBuildEnvironment (ceconan libs)
      |    |- prepareArguments()  = optionsForFilter + optionsForBackend + compiler.options
      |    |                        + library flags + filterUserOptions + orderArguments
      |    |- Promise.all: runCompiler() -> this.exec() -> lib/exec.ts
      |    |               generateAST / generatePP / generateIR / generateOptPipeline / "independent" tools
      |    '- checkOutputFileAndDoPostProcess -> postProcess -> external parser (objdump+parse),
      |                                                       or objdump (binary) / read asm file
      '- afterCompilation()
           |- handleExecution() started early (doExecution -> getOrBuildExecutable -> runExecutable, or remote)
           |- "postcompilation" tools
           |- processAsm()  -> this.asm.process (TS AsmParser) or LlvmIrParser; skipped if externalParserUsed
           |- postProcessAsm() -> demangler
           |- cfg.generateStructure (if requested)
           '- cleanupResult -> env.cachePut -> storeOversizedResult (S3 when large)
```

CMake/IDE-mode builds go through `buildProject` / `cmake` -> `afterCmakeCompilation`.

Most-overridden hooks in `lib/compilers/*` (rough counts): `optionsForFilter` (130), `getArgumentParserClass` (70),
`getOutputFilename` (54), `getSharedLibraryPathsAsArguments` (52), `runCompiler` (49), `processAsm` (29),
`isCfgCompiler` (27), `orderArguments` (22), `getDefaultExecOptions` (19), `filterUserOptions` (18), `objdump` (17).
`compile`/`doCompilation` are essentially never overridden. Subclasses also commonly reassign `this.asm` in their
constructor to swap the assembly parser.

The compiler's identity comes from `CompilerInfo` (`types/compiler.interfaces.ts`), built from properties by
`CompilerFinder.compilerConfigFor`. `initialise()` probes the version (`versionFlag`/`versionRe`); a compiler whose
version cannot be read is silently dropped.

### 3.4 Environment, queue, caches, temp

* `lib/compilation-env.ts` `CompilationEnvironment`: holds three caches built from `cacheConfig`,
  `executableCacheConfig`, `compilerCacheConfig` (`lib/cache/from-config.ts` parses `InMemory(n)`, `OnDisk(path,n)`,
  `S3(bucket,path,region)`, `;`-joined into a `MultiCache`); `optionsAllowedRe`/`optionsForbiddenRe`; the env passed to
  child processes; `enqueue`.
* `lib/compilation-queue.ts`: a `p-queue` with `maxConcurrentCompiles` concurrency and `compilationStaleAfterMs`
  staleness. A nested `enqueue` from inside a job runs inline (deadlock guard), so don't assume queue isolation.
* `lib/temp.ts`: temp dirs under `TMPDIR`; `CompileHandler` sweeps them every `tempDirCleanupSecs` (default 30).
* Compilation workers (`compilequeue.is_worker`) require a shared S3 cache layer or startup throws.

### 3.5 Running processes and sandboxing (`lib/exec.ts`)

Two independent knobs, both in `etc/config/execution.*.properties`, both taking `none | nsjail | firejail | cewrapper`:

* `executionType`: how compilers, tools, demanglers are run (`execute()`, `executeDispatchTable`).
* `sandboxType`: how user binaries are run (`sandbox()`, `sandboxDispatchTable`).

`execution.defaults` sets both to `none`. Every Linux production environment (`amazon`, `gpu`, `aarch64prod`,
`aarch64staging`) sets both to `nsjail` with `etc/nsjail/*.cfg` (`docs/NsjailSandbox.md`), and that is the only
sandbox actively maintained. `firejail` is the legacy predecessor: its profiles are still in `etc/firejail/`, but
the production files explicitly blank `firejail=` and nothing new should target it. `cewrapper` is the Windows
equivalent, used only by `execution.amazonwin`. Timeouts: `compileTimeoutMs` (compilers), `binaryExecTimeoutMs`
(user code). Output is truncated at `maxOutput` (1 MiB default). On timeout the result is marked
`okToCache=false`.

Execution environments (`lib/execution/`): `LocalExecutionEnvironment`, `DotnetExecutionEnvironment`,
`RemoteExecutionEnvironment` (SQS + WebSocket, used when the execution triple `iset-os-specialty` doesn't match the
host). Worker-mode details are in AGENTS.md "Worker Mode Configuration".

### 3.6 Keyed registries (the plugin pattern)

Every pluggable family follows one shape (`lib/keyed-type.ts`): each class has `static get key()` (string or array),
`<dir>/_all.ts` re-exports every class, `<dir>/index.ts` calls `makeKeyedTypeGetter`. Duplicate keys throw at import
time; unknown keys throw `No <kind> named '<key>' found` at use time (an unknown `compilerType` exits the process).

| Directory | Getter | Selected by property |
|---|---|---|
| `lib/compilers` | `getCompilerTypeByKey` | `compilerType` (empty -> `default`) |
| `lib/demangler` | `getDemanglerTypeByKey` | `demanglerType` |
| `lib/objdumper` | `getObjdumperTypeByKey` | `objdumperType` |
| `lib/formatters` | `getFormatterTypeByKey` | `formatter.X.type` |
| `lib/tooling` | `getToolTypeByKey` | `tools.X.class` |
| `lib/buildenvsetup` | `getBuildEnvTypeByKey` | `buildenvsetup` (e.g. `ceconan`) |
| `lib/external-parsers` | `getExternalParserByKey` | `externalparser` (`CEAsmParser`, `plain`) |
| `lib/execution` | `getExecutionEnvironmentByKey` | chosen by compiler class |
| `lib/storage` | `getStorageTypeByKey` | `storageSolution` (`local`/`s3`/`remote`/`null`) |
| `lib/shortener` | `getShortenerTypeByKey` | `urlShortenService` |
| `lib/asm-docs` | `getDocumentationProviderTypeByKey` | `instructionSet` |
| `lib/cfg/cfg-parsers` | `getParserByKey` (defaulted) | compiler group |
| `lib/cfg/instruction-sets` | `getInstructionSetByKey` (defaulted) | `instructionSet` |

`lib/build-systems` is a static map instead (`shared/build-systems.ts` holds the descriptors). `lib/runtime-tools`
has an `_all.ts` but no keyed getter.

### 3.7 Output processing

* Two assembly parsers exist and must stay in step. The TypeScript one is `lib/parsers/asm-parser.ts`:
  `AsmParser.process` applies the filters (directives, labels, comments, library code, binary) and maps asm
  lines back to source lines. Specialised subclasses live beside it (`asm-parser-vc.ts`, `-ptx`, `-spirv`, ...)
  and are picked by compiler constructors. It handles every compiler that has no `externalparser`, which means
  all local/dev installs and the long tail of languages in production.
* The C++ one lives in the sibling [asm-parser](https://github.com/compiler-explorer/asm-parser) repo and is a
  faster reimplementation of the same filtering. `lib/external-parsers/` drives it: `ExternalParserBase` maps
  filters to CLI flags (`-unused_labels`, `-directives`, `-comment_only`, `-whitespace`, `-binary`, `-plt`), runs
  objdump itself in binary mode, and sets `externalParserUsed` so `afterCompilation` skips `processAsm`. It is
  enabled per language with `externalparser=CEAsmParser` and `externalparser.exe=/usr/local/bin/asm-parser`;
  production turns it on for c, c++, objc, objc++, rust, gimple, v, vala, jakt and modula2. The `plain` key is a
  generic variant (micropython's `mpy-tool.py`). Never used for remote compilers.
* `lib/llvm-ir.ts`, `lib/llvm-ast.ts`, `lib/parsers/*-pass-dump-parser.ts`: IR, AST, opt-pipeline views.
* `lib/demangler/*`: run the demangler binary over collected symbols. `lib/objdumper/*`: disassemble binaries.
* `lib/cfg/`: control-flow graphs; parser chosen per compiler family, ISA per `instructionSet`.
* Result shape: `CompilationResult` in `types/compilation/compilation.interfaces.ts`. New per-view outputs are added
  there as `<x>Output`, requested via `compilerOptions.produce<X>`, gated by `CompilerInfo.supports<X>`.

### 3.8 State persistence and shortlinks

* `lib/storage/base.ts` `StorageBase.handler`: hash the config (profanity-checked, nonce retry), find a unique
  sub-hash, store, return `/z/<id>`. Backends: `local.ts` (`localStorageFolder`), `s3.ts` (S3 + DynamoDB),
  `remote.ts` (proxy), `null.ts`.
* `lib/clientstate.ts` is the public, stable "ClientState" schema used by `/clientstate/` and the API.
  `lib/clientstate-normalizer.ts` converts both ways: `ClientStateNormalizer` (GoldenLayout -> ClientState),
  `ClientStateGoldenifier` and `configToGoldenLayout` (ClientState -> GoldenLayout). Adding a pane type means adding
  cases there too.
* Never rename GoldenLayout component names or state keys: existing shortlinks store them.

### 3.9 Cross-cutting

`lib/logger.ts` (winston; `suppressConsoleLog` in tests), `lib/sentry.ts`, `lib/stats.ts` (compilation stats to S3),
Prometheus counters via `prom-client` served by `lib/metrics-server.ts`, `lib/assert.ts` (`assert`, `unwrap`),
`lib/utils.ts` (`splitLines`, `parseOutput`, `maskRootdir`, `anonymizeIp`, `resolvePathFromAppRoot`), `lib/aws.ts`
(SSM-backed config), `lib/csp.ts` (policy string only; the `csp` middleware is a no-op).

## 4. Configuration system (`etc/config/*.properties`)

Authoritative detail: `docs/Configuration.md`, `docs/AddingACompiler.md`. Loader: `lib/properties.ts`. Consumer:
`lib/compiler-finder.ts`.

### 4.1 Files and hierarchy

Files are `<base>.<level>.properties`. `<base>` is a language id (`c++`, `rust`) or one of `compiler-explorer`,
`execution`, `aws`, `asm-docs`, `builtin`. `<level>` is a hierarchy level. `createPropertyHierarchy` in
`lib/app/config.ts` builds, lowest to highest priority:

```
defaults -> <env>... (CLI order) -> <env>.<platform>... -> <platform> -> <hostname> -> local (unless --no-local)
```

Resolution is **per key**: the highest level that defines a key wins; files never replace each other wholesale.
`defaults` is always loaded, so a key in `c++.defaults` leaks into `--env amazon` unless amazon overrides it. Group
keys (`group.G.*`) resolve through the same hierarchy as any other key. There are no `*.dev.properties`; dev is
effectively defaults + platform + hostname + local. `*.local.properties` are gitignored and are how a developer points
CE at their own compilers (or use `etc/scripts/ce-properties-wizard`).

### 4.2 Per-compiler lookup chain

For property `prop` on compiler `X` in group `G` (nested in `P`), `CompilerProps`/`compilerConfigFor` try in order:
`compiler.X.prop` -> `group.G.prop` -> `group.P.prop` -> bare `prop` in the language file -> `prop` in
`compiler-explorer.*` -> code default. Each step scans the full hierarchy, so **specific beats general regardless of
level**: `compiler.X.options` in `c++.defaults` beats `group.G.options` in `c++.local`.

### 4.3 Syntax essentials

* `key=value`; `#` starts a comment anywhere on the line (values cannot contain `#`).
* `key+=value` appends to a string key already defined earlier **in the same file**.
* Booleans/numbers are coerced; keys ending `.version`/`.semver` stay strings.
* `:` separates ids (`compilers=gcc:&clang-group:host@port/base`), `|` separates argument lists
  (`versionFlag`, `demanglerArgs`, `postProcess`), `path.delimiter` separates path lists.
* `compilers=` is one key: a higher level replaces the whole list. Group membership comes only from
  `group.G.compilers`. `&G` recurses into a group; `host@port/base` pulls a remote CE instance's compiler list
  (needs remote fetch enabled; `--no-remote-fetch` disables).

### 4.4 Properties that matter most

Compiler-level (`types/compiler.interfaces.ts` `CompilerInfo`): `exe`, `name`, `compilerType`, `options`,
`versionFlag`/`versionRe`/`explicitVersion`, `semver`/`isSemVer`/`baseName`, `alias` (old ids that must keep
resolving for shortlinks), `instructionSet` (must be in `InstructionSetsList`), `demangler`/`demanglerType`,
`objdumper`/`objdumperType`, `supportsBinary`, `supportsBinaryObject`, `supportsExecute`, `interpreted`,
`executionWrapper`, `intelAsm`, `includeFlag`/`linkFlag`/`libpathFlag`/`rpathFlag`, `libPath`/`ldPath`/`extraPath`/
`envVars`, `buildenvsetup`, `externalparser`, `postProcess`, `unwiseOptions`, `hidden`, `notification`,
`license*`. Language-level: `compilers`, `defaultCompiler`, `group.X.*`, `libs`/`libs.X.*`, `tools`/`tools.X.*`,
`formatters`.

Server-level (`compiler-explorer.defaults.properties`): `compileTimeoutMs`, `binaryExecTimeoutMs`,
`maxConcurrentCompiles`, `cacheConfig`/`executableCacheConfig`/`compilerCacheConfig`, `storageSolution`,
`localStorageFolder`, `urlShortenService`, `optionsAllowedRe`/`optionsForbiddenRe`, `maxUploadSize`, `trustProxy`,
`textBanner`, `cookiePolicyEnabled`/`privacyPolicyEnabled`, `ceToolsPath`, `restrictToLanguages`,
`rescanCompilerSecs`, `compilequeue.*`/`execqueue.*`.

The browser never sees `exe`, `compilerType`, `versionFlag`, `demangler`, `objdumper` and similar keys;
`ClientOptionsHandler.setCompilers` strips them.

### 4.5 Validation

`lib/properties-validator.ts` runs over every real config file in `test/properties-validation-tests.ts`
(`npm run test:props`; local files only with `CHECK_LOCAL_PROPS=true`). It checks duplicate keys, empty list elements,
orphaned `compiler.X.exe`/ids/groups/libs/tools, invalid `defaultCompiler`, suspicious exe paths outside
`/opt/compiler-explorer` in amazon files, and duplicate compiler ids across amazon files. "Listed" means reachable
from `compilers=`/`group.X.compilers` **in the same file**. Suppress with a `# Disabled: id1 id2` comment.
PRs touching Conan-built language amazon files also get `etc/scripts/check_infra_settings.py` run in CI against the
infra repo's `settings.yml`.

## 5. Frontend (`static/`)

### 5.1 Stack and hard constraints

jQuery 4, Bootstrap 5 (`data-bs-*`), GoldenLayout 1.x (CE fork, legacy jQuery API), Monaco, tom-select, underscore,
lz-string. TypeScript compiled to **ES5** (`tsconfig.frontend.json`; `lib: dom, es5, dom.iterable`), bundled by
webpack 5.

* Relative imports end in `.js` even though files are `.ts` (webpack `extensionAlias`; same rule server-side for ESM).
* `static/` must not import from `lib/`. Enforced by `etc/scripts/check-frontend-imports.js` (a `git grep` for
  `from '../...lib/'`). Share via `types/` or `shared/`.
* No component has a reference to another; everything goes through the event bus (section 5.3).
* Every piece of pane state ends up in the URL, so keep state small and never rename keys.

### 5.2 Boot (`static/main.ts`)

`options.ts` reads `div#config` (injected by `views/_layout.pug` from `lib/app/rendering.ts` `renderConfig`) into
`window.compilerExplorerOptions`. Then `start()`: load languages (`static/services/languages.service.ts`), pick the
layout via `findConfig()` in priority order embedded params -> `options.slides` -> `options.config` (server-injected
for `/z/`) -> `location.hash` -> session/local storage `gl` -> default (editor + compiler). Then
`new GoldenLayout(...)`, `new Hub(layout, ...)`, `setupSettings`, `hub.initLayout()`, sharing, history, MOTD.

### 5.3 Hub and the event bus

* `static/hub.ts` `Hub`: registers every pane component with GoldenLayout (`registerComponent` + a factory), hands out
  editor/compiler/executor/tree ids, holds the single `CompilerService`. Emissions before `initLayout()` are deferred
  and replayed, `compiler` events last.
* `static/event-hub.ts` `EventHub`: per-pane wrapper around GoldenLayout's bus; `unsubscribe()` on close.
* `static/event-map.ts` `EventMap`: hand-maintained typed map of every event name and signature. Add new events here.

Key events:

| Event | Emitter -> consumers |
|---|---|
| `editorChange(editorId, source, langId)` | Editor (debounced by the `delayAfterChange` setting, 2000ms default) -> Compiler, Executor, Tree |
| `requestCompilation` | Editor/Tree (manual trigger) -> Compiler, Executor |
| `compiler(compilerId, compilerInfo, options, editorId, treeId)` | Compiler announces itself -> views, Editor, Diff |
| `compiling`, `compileResult(compilerId, compiler, result, lang)` | Compiler -> every view (filter by `compilerId`), Editor markers, Output |
| `compilerOpen`/`compilerClose`, `editorOpen`/`editorClose`, `treeOpen`/`treeClose`, `executorOpen` | lifecycle; `compilerClose` auto-closes dependent panes |
| `<x>ViewOpened`/`<x>ViewClosed(compilerId)` | view -> Compiler flips `produce<X>` and recompiles |
| `<x>ViewOptionsUpdated` | view -> Compiler (pp, ir, optPipeline, ...) |
| `languageChange`, `findCompilers`/`findEditors`, `resendCompilation`, `compilerFlagsChange` | discovery and re-sync |
| `coloursForCompiler`/`coloursForEditor`, `panesLinkLine`, `editorLinkLine` | source/asm line colouring and hover linking |
| `settingsChange`, `requestSettings`, `themeChange`, `broadcastFontScale`, `renamePane`, `resize`, `shown` | global UI |

### 5.4 Pane lifecycle (`static/panes/pane.ts`)

`Pane<State>` constructor: `getInitialHTML()` (clones `$('#<templateId>').html()` from hidden templates rendered by
`views/templates/templates.pug`) -> `initializeCompilerInfo` -> `initializeDefaults` -> settings -> state-dependent
props -> `registerDynamicElements` -> `registerButtons` -> `registerStandardCallbacks` (rename, destroy -> `close`,
resize, `compileResult` -> `onCompileResult`, `compiler` -> `onCompiler`, `compilerClose`, `settingsChange`) ->
`registerCallbacks`. `getCurrentState()`/`updateState()` feed `container.setState`, which is what gets serialised.
`MonacoPane` adds a Monaco editor in `.monaco-placeholder`, font scaling, print support.

Big panes: `compiler.ts` (asm output, filters, libs, overrides, tools, all the view buttons), `editor.ts` (source,
language switching, diagnostics), `tree.ts` (IDE mode; not a `Pane`, owns `MultifileService`), `executor.ts`,
`output.ts`, `diff.ts`, `conformance-view.ts`, `cfg-view.ts`. The many `*-view.ts` panes (ir, opt, ast, pp, rustmir,
stack-usage, ...) all follow one pattern: extend `MonacoPane`, emit `<x>ViewOpened` on construct, filter
`compileResult` by id and read `result.<x>Output`, toggle "not supported" from `compiler.supports<X>`, emit
`<x>ViewClosed` in `close()`.

### 5.5 Browser-side compile flow

Editor debounces -> `editorChange` -> `Compiler.onEditorChange` -> `compile()` builds the request
(`userArguments`, `compilerOptions.produce*`, `filters`, `tools`, `libraries`, `executeParameters`) ->
`sendCompile()` (one in flight, latest queued request wins) -> `CompilerService.submit` (`static/compiler-service.ts`;
small LRU keyed by request JSON, honours `bypassCache`/`okToCache`) -> `$.ajax POST api/compiler/:id/compile` ->
`onCompileResponse` -> emit `compileResult`. CMake builds use `submitBuild`; `#include <http...>` is expanded
client-side via `download-service.ts`.

### 5.6 State, URLs, settings

* `shared/url-serialization.ts`: `serialiseState` = minify GoldenLayout config (positional base36 keys; **append,
  never reorder** in `ConfigMinifier`) -> rison -> optional lz-string `{z:...}`. `static/url.ts` `loadState` upgrades
  v1 to v3 states.
* `#<state>` lives client-side; `/z/:id` is server-stored; `/clientstate/<b64>` is the API schema inline.
* `static/sharing.ts`: full/short/embed links; `POST api/shortener`. Replaces a stale `/z/` URL in the address bar
  after the layout settles. `selection` is stripped from shared state.
* `static/settings.ts` `SiteSettings`: add a field, an entry in the matching `add*` list, and a control in
  `views/popups/settings.pug`. Stored in localStorage under `options.localStoragePrefix` (`static/local.ts`).
* `static/history.ts`: last 30 layouts in localStorage.

### 5.7 Templates, styles, build

* `views/_layout.pug` -> `index.pug`/`embed.pug`; `views/templates/panes/*.pug` and the `+monacopane("id")` mixin in
  `views/templates/templates.pug` render hidden `.gl_keep.template` divs; `views/popups/*.pug` are modals.
* `static/styles/{explorer,colours}.scss` plus `styles/themes/`; themes in `static/themes.ts` (`Themer`).
* `webpack.config.esm.ts`: entries `main` and `noscript`; output `out/webpack/static` with content hashes and a
  manifest at `out/dist/manifest.json`; `MonacoEditorWebpackPlugin` with an explicit language list;
  `etc/webpack/parsed-pug-loader.js` turns `static/generated/*.pug` (policies, changelog) into `{hash, text}`;
  Terser `ecma: 5`. Dev mode serves through webpack-dev-middleware with HMR.

## 6. Shared code (`types/`, `shared/`)

`types/compiler.interfaces.ts` (`CompilerInfo`), `types/compilation/compilation.interfaces.ts`
(`CompilationRequest`, `CompilationResult`, `ExecutionOptions`, cache keys), `types/languages.interfaces.ts`
(`LanguageKey` union; languages themselves are hardcoded in `lib/languages.ts`), `types/features/filters.interfaces.ts`
(`ParseFiltersAndOutputOptions`), `types/tool.interfaces.ts`, `types/libraries/*`. `shared/` holds
`url-serialization.ts`, `rison.ts`, `build-systems.ts`, `common-utils.ts`, `remote-utils.ts`, `assert.ts`.

## 7. Testing map

Vitest with two projects (`vitest.config.ts`): `unit` (`test/**/*.ts`, every `.ts` file is collected except `_*.ts`
and `utils.ts`; setup files fake AWS creds and silence logs) and `frontend unit` (`static/tests/**/*.ts` under
happy-dom; `_setup-dom.ts` injects a `#config` div). Run one project with `npx vitest run --project unit` or
`--project 'frontend unit'`. Convention: `*-tests.ts`.

Helpers in `test/utils.ts`: `makeCompilationEnvironment({languages, props, doCache, ...})`,
`makeFakeCompilerInfo`, `makeFakeLanguage`, `makeFakeParseFiltersAndOutputOptions`, `shouldExist`, `newTempDir`,
`resolvePathFromTestRoot`, `processAsm(filename, filters)` (picks a parser from the filename prefix),
`skipExpensiveTests`.

Patterns:

* **Compiler class**: `new FooCompiler(makeFakeCompilerInfo({exe:'/dev/null', lang:'foo'}), ce)`, then
  `vi.spyOn(compiler, 'exec').mockResolvedValue({code:0, stdout:'', stderr:''})` and call `postProcess`,
  `optionsForFilter`, etc. Examples: `test/golang-tests.ts`, `test/mach-tests.ts`, `test/compilers/*`.
* **Handler**: build a tiny Express app with supertest; seed `CompileHandler.setCompilers([{compilerType:
  'fake-for-test', fakeResult: {...}}])` (`lib/compilers/fake-for-test.ts`). Example: `test/handlers/compile-tests.ts`.
* **Asm filter goldens**: drop `foo.asm` in `test/filters-cases/` (prefix `ptx-`, `sass-`, `ca65-`, ... picks the
  parser; `-bin.asm` adds binary variants) and run `npx vitest run -u test/filter-tests.ts` to write
  `foo.asm.*.json` via `toMatchFileSnapshot`. Same `-u` flow for `test/demangle-cases/`. These suites are gated by
  `SKIP_EXPENSIVE_TESTS=true` (`npm run test-min`) and skipped on Windows/macOS.
* **Other fixtures**: `test/cfg-cases/*.json`, `test/state/` (layout normalisation), `test/example-config/`
  (hierarchy), `test/test-properties/`.
* **Cypress** (`cypress/e2e/*.cy.ts`): start `npm run dev -- --language c++ --no-local`, then `npm run cypress`.
  `cypress/e2e/frontend.cy.ts` has `PANE_DATA_MAP`; add new panes there. Guidance in `docs/internal/FrontendTesting.md`.

## 8. Build, run, CI

* `make dev` (tsx watch on `.ts` and `etc/config/*`, installs deps, resolves Node via `etc/scripts/find-node` and
  `.node-version`), `make gpu-dev` (`--env gpu`), `make debug` (inspector), `make` / `make run` (production-style:
  webpack + tsc to `out/`, then `node out/dist/app.js --static out/webpack/static`). Pass server flags with
  `make EXTRA_ARGS='--language c++' dev`. `npm run dev` is the same without watching or dependency install.
  Port 10240.
* `npm run check` = ts-check (5 tsconfig projects) + biome lint-check + frontend-import check + license-header check
  + `test-min`. Pre-commit (`.husky/pre-commit` + `lint-staged.config.mjs`) runs biome fix, full ts-check and
  `vitest related` on staged `.ts`, `test:props` on `.properties`, then the two scripts. New `.ts`/`.js` files need the
  BSD-2 banner (copy from any file and fix the year).
* CI (`.github/workflows/`): `test-and-deploy.yml` (lint, license, full tests with coverage, ts-check, Python wrapper
  tests; on push to the org repo builds a dist tarball, uploads to S3 and tags `gh-<run>`), `test-win.yml`,
  `test-frontend.yml` (Cypress, push only), `check-infra-settings.yml`, CodeQL, actionlint. Releases to godbolt.org
  are driven from the infra repo's `ce` tool, not from here.

## 9. Common tasks: where to look

| Task | Touch | Reference |
|---|---|---|
| Add a compiler that an existing class handles | `etc/config/<lang>.{local,amazon}.properties` (use the wizard locally) | `docs/AddingACompiler.md`, `.claude/agents/compiler-config.md` |
| New compiler class | `lib/compilers/foo.ts` with `static get key()`, export from `lib/compilers/_all.ts`, `compilerType=foo` | sections 3.3, 3.6 |
| New language | `lib/languages.ts` + `LanguageKey` in `types/languages.interfaces.ts`, `etc/config/<id>.*.properties`, `examples/<id>/default.<ext>`, `static/modes/<id>-mode.ts` + `modes/_all.ts`, logo | `docs/AddingALanguage.md` |
| New output pane | `event-map.ts`, `components.interfaces.ts`, `components.ts` (incl. `validateComponentState`), `hub.ts`, `panes/<x>-view.ts`, template in `views/templates/`, button in `views/templates/panes/compiler.pug`, wiring in `panes/compiler.ts`, `supports<X>` in `services/compilers.service.ts`, backend `produce<X>`/`<x>Output`, `lib/clientstate-normalizer.ts`, `static/tests/url-tests.ts`, Cypress `PANE_DATA_MAP` | copy `static/panes/stack-usage-view.ts` |
| New tool / formatter / demangler / objdumper / storage | class + `static get key()` + `_all.ts` export + property | `docs/AddingATool.md`, `docs/AddingAFormatter.md` |
| New library | `libs.X.*` properties (+ infra install) | `docs/AddingALibrary.md`, `docs/AboutLibraryPaths.md` |
| New REST endpoint | controller in `lib/handlers/api/`, register in `lib/app/controllers.ts`/`routes-setup.ts`, document in `docs/API.md`, test with supertest | section 3.2 |
| New user setting | `static/settings.ts` + `views/popups/settings.pug` | section 5.6 |
| Change asm filtering | `lib/parsers/asm-parser.ts`, regenerate `test/filters-cases` goldens, and mirror it in the C++ asm-parser repo or prod output will differ | sections 3.7, 7 |
| New server property | read via `ceProps(...)`/`compilerProps(...)`, default in `compiler-explorer.defaults.properties`, mention in `docs/Configuration.md` | section 4 |
| Assembly docs for an ISA | `etc/scripts/docenizers` -> `lib/asm-docs/generated/` (`make asm-docs`) | `docs/AddingAssemblyDocumentation.md` |

## 10. Gotchas (consolidated)

1. ESM everywhere: import `./foo.js` for `./foo.ts`, on both sides.
2. Registration is `static get key()` + `_all.ts`. Forgetting the export gives a runtime "No X named" error.
3. `executionType` (compilers/tools) is not `sandboxType` (user binaries). Production is nsjail on Linux and
   cewrapper on Windows; firejail is legacy. The code fallback for a missing `sandboxType` is still `firejail`,
   while the shipped default is `none`, so set both explicitly.
4. Property resolution is per key, `defaults` always loads, specific-beats-general across levels, `compilers=` is
   replaced wholesale, `+=` is same-file only, `#` cannot appear in values.
5. Silent drops: compilers with no detectable version, tools whose exe is missing. Loud failures: unknown
   `compilerType`, duplicate registry keys.
6. A `@` in a `compilers=` entry means "remote CE instance"; requests are proxied over HTTP, not compiled locally.
7. `req.body` on the compile route can be an object or a string. Compile *failures* still return HTTP 200 with
   `code: -1`; only malformed requests get 400.
8. Nested `CompilationQueue.enqueue` runs inline. Compilation workers need an S3 cache layer.
9. Frontend: ES5 target, no `lib/` imports, events before `initLayout()` are deferred, `compilerClose` cascades to
   dependent panes, `ConfigMinifier` keys are positional, and the component-name constants in
   `static/components.interfaces.ts` are frozen because shortlinks store them (the opt-pipeline view is still
   `llvmOptPipelineView`).
10. Every `.ts` under `test/` is a test. Golden files regenerate with `vitest -u`, which is otherwise undocumented.
11. `check-frontend-imports` is not run in CI, only by pre-commit and `npm run check`.
12. `lib/base-compiler.ts` and `static/panes/compiler.ts` are huge; grep symbols, don't trust remembered offsets.
13. Prod runs as a cluster with no shared in-process state. Anything that must survive a request goes in the cache,
    storage, or the URL.
14. On godbolt.org, C, C++ and Rust assembly is filtered by the external C++ `asm-parser` binary, not by
    `lib/parsers/asm-parser.ts`. A fix in the TypeScript parser alone changes dev output but not production.

## 11. Glossary

* **Filters**: the asm post-processing toggles (`binary`, `binaryObject`, `execute`, `intel`, `demangle`, `labels`,
  `directives`, `commentOnly`, `libraryCode`, `trim`, `debugCalls`); see `ParseFiltersAndOutputOptions`.
* **Overrides**: user-selected compiler-level switches (stdver, arch, env vars) sent as `compilerOptions.overrides`.
* **Runtime tools**: wrappers around user-binary execution (heaptrack, libsegfault).
* **Tools**: external programs run on the source or output (clang-tidy, llvm-mca, pahole, ...), `lib/tooling/`.
* **Tree / IDE mode**: multi-file projects with CMake/Cargo/Maven/Make drivers (`docs/IDEMode.md`,
  `docs/BuildSystems.md`).
* **Shortlink**: `/z/<id>`, stored server-side. **ClientState**: the stable JSON schema for `/clientstate/` and the
  API. **GoldenLayout config**: the raw layout the browser uses; the two convert via `clientstate-normalizer.ts`.
* **Prediscovered**: a JSON dump of compiler metadata produced by `--discoveryonly` and consumed by
  `--prediscovered`, so prod instances skip probing thousands of binaries at boot.
* **Execution triple**: `iset-os-specialty` describing where a binary can run; mismatches go to remote SQS workers.
* **Conan / ceconan**: prebuilt library binaries fetched at link time by `lib/buildenvsetup/ceconan.ts`.
* **MOTD**: message of the day / sponsor banner (`static/motd.ts`).

## 12. Related repositories and docs

* [compiler-explorer/infra](https://github.com/compiler-explorer/infra): installs compilers and libraries into
  `/opt/compiler-explorer`, builds Conan libraries, deploys and releases (`ce` tool). Often checked out at `../infra`.
* [compiler-explorer-tools](https://github.com/compiler-explorer/compiler-explorer-tools): demanglers and helpers
  referenced by `ceToolsPath`.
* [asm-parser](https://github.com/compiler-explorer/asm-parser): the C++ assembly filter used in production for
  the mainstream languages (section 3.7).
* Start-here docs: `docs/Configuration.md`, `docs/API.md`, `docs/AddingACompiler.md`, `docs/AddingALanguage.md`,
  `docs/Privacy.md`, `docs/NsjailSandbox.md`, `docs/internal/FrontendTesting.md`, `docs/VitestCribSheet.md`.
