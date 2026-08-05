# Build systems

Compiler Explorer's IDE ("tree") mode can hand a whole project to a build system rather than invoking a compiler on a
single source file. The only supported build system is CMake, and it is wired in as a special case at nearly every
layer: a boolean on the tree state, a dedicated API route, and a boolean on the compilation queue message.

This document describes how that works, and the incremental plan for turning it into a pluggable mechanism so that
Cargo (Rust), Maven/Gradle (Java, Kotlin) and others can be added without another round of special-casing. Phase 1
below has landed; the rest has not.

## How CMake works

### Backend

`BaseCompiler.buildProject()` (`lib/base-compiler.ts`) orchestrates the flow, with the CMake-specific parts supplied by
`CMakeBuildSystem` (`lib/build-systems/cmake.ts`). `BaseCompiler.cmake()` is a thin delegate onto it.

1. Refuse if `!compiler.supportsBinary`; force `filters.binary` and `filters.dontMaskFilenames`
   (`getUnsupportedReason()`, `applyRequestDefaults()`).
2. Create a temp dir, write the main source as `CMakeLists.txt` plus the extra files (`writeProjectFiles()`), and
   `mkdir build/` (`prepareBuildDirectory()`).
3. Build a cache key (`getBuildProjectCacheKey()`) discriminated by `api: 'cmake'`, and try to load a prebuilt
   executable package from the cache before doing any work.
4. Run the plan's steps via `doBuildstepAndAddToResult()`: the configure step (`ceProps('cmake')`, optionally `-GNinja`
   from `ceProps('useninja')`, plus the toolchain param and user `cmakeArgs`), then `cmake --build .`.
5. Locate the artifact by convention: `getExecutableFilename(dirPath/build, outputFilebase, key)`, i.e. `build/output.s`
   unless `backendOptions.customOutputFilename` overrides it.
6. Disassemble it (`checkOutputFileAndDoPostProcess()`), optionally execute it locally or remotely
   (`RemoteExecutionQuery`), then run `afterCmakeCompilation()` (tools, opt/stack-usage output, caching) and clean up.

The compiler to use is injected through environment variables rather than a command line:

- `getCmakeBaseEnv()` sets `CXX`/`CC` (C++), `FC` (Fortran), `CUDACXX` (CUDA), `AS` (assembly), `CC` otherwise.
- `getCompilerEnvironmentVariables()` sets `CXXFLAGS`/`FFLAGS`/`CUDAFLAGS`/`CFLAGS`.
- `createCmakeExecParams()` sets `LDFLAGS` and `ldPath`.

Compilers that need to deviate override `getExtraCMakeArgs()` (`win32-vc`, `llvm-mos`), `getCMakeExtToolchainParam()`
(`llvm-mos`) or the environment (`cc65`). These hooks live on `BaseCompiler` and are called by `CMakeBuildSystem`.

Note the configure and build steps deliberately share one `env` object — `createCmakeExecParams()` takes a shallow copy
— which is how the build step sees `CXXFLAGS` at all.

Surrounding plumbing:

| Concern            | Where                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------- |
| HTTP route         | `POST /api/compiler/:id/cmake` → `CompileHandler.handleCmake()` (`lib/handlers/compile.ts`) |
| Sub-server proxying| `compilerInfo.cmakePath` (`types/compiler.interfaces.ts`, set in `lib/compiler-finder.ts`) |
| Queue worker       | `RemoteCompilationRequest.isCMake` boolean (`lib/compilation/sqs-compilation-queue.ts`) — produced outside this repo |
| Cache key          | `CmakeCacheKey`'s `api: 'cmake'` discriminator (`types/compilation/compilation.interfaces.ts`) |
| Stats              | `KnownBuildMethod.CMake` (`lib/stats.ts`)                                              |
| Metrics            | `ce_cmake_compilations_total`, `ce_cmake_executions_total`, and SQS equivalents        |
| Config             | `cmake=` and `useninja=` in `etc/config/compiler-explorer.*.properties`                |

### Frontend

- `MultifileService` holds `isCMakeProject: boolean`; `isCompatibleWithCMake()` hardcodes the language list
  (c++, c, fortran, cuda, assembly); the main source file is `CMakeLists.txt` when the flag is set.
- The tree pane (`static/panes/tree.ts`) exposes an on/off toggle plus `cmakeArgs` and `customOutputFilename` inputs,
  all part of `TreeState`.
- The compiler and executor panes each duplicate a parallel request path — `sendCMakeCompile()`,
  `pendingCMakeRequestSentAt`, `nextCMakeRequest`, `onCMakeResponse()` — chosen by `isACMakeProject()`.
- `CompilerService.submitCMake()` posts to the `/cmake` URL.
- `cmake` is a pseudo-language in `lib/languages.ts`, force-exposed to the frontend even when no compiler claims it
  (`lib/handlers/api.ts`, `lib/app/config.ts`) so tree mode can always resolve it.
- Shortlinks persist the flag through `ClientStateTree` (`lib/clientstate.ts`); `test/state/*.json` are golden fixtures
  of that format.

## Why this doesn't generalise

The plumbing is the easy part. The hard part is four assumptions baked into `cmake()`:

1. **The compiler is selected via C-style environment variables.** Cargo wants `RUSTC`/`CARGO`; Maven wants a JDK on
   `JAVA_HOME`.
2. **The artifact lives at a conventional path.** Cargo needs `--message-format=json` (or `cargo metadata`) to discover
   `target/debug/<name>`; Maven produces `target/*.jar` named from the POM.
3. **"Output" means disassembling a native binary.** For Cargo the useful output is `cargo rustc -- --emit asm`; for
   Maven it is javap-style bytecode, and there is no native binary at all.
4. **Execution means running the artifact directly.** Maven artifacts need `java -jar`.

So the abstraction has to cover manifest handling, build steps, environment, artifact discovery, post-processing and
execution shape — not just "which binary do we run".

## Plan

Phases 0–2 are refactoring with no user-visible change. Each is independently shippable.

### Phase 0 — protocol groundwork

Add `shared/build-systems.ts` with a `BuildSystemId` type (initially just `'cmake'`) and descriptors both frontend and
backend need: display name, manifest filename, manifest language id, compatible language ids, default arguments,
argument-input placeholder.

Widen the wire in back-compatible ways:

- New route `POST /api/compiler/:id/build/:buildSystem`, with `/cmake` kept **permanently** as an alias (it is
  documented public API in `docs/API.md`).
- `backendOptions.buildSystemArgs`, accepting `cmakeArgs` as a fallback.
- `CmakeCacheKey.api` becomes `BuildSystemId`, keeping the literal `'cmake'` so existing cache and executable-package
  hashes stay valid.
- `RemoteCompilationRequest.buildSystem?: BuildSystemId` alongside a still-honoured `isCMake` — the producer of those
  messages lives in another repository, so both must work across a rolling deploy.
- `compilerInfo.buildPath` alongside `cmakePath`, for the same reason.

### Phase 1 — extract a build system driver (done)

`BaseCompiler.cmake()` is now a one-line delegate to a generic
`buildProject(buildSystem, files, parsedRequest, bypassCache)` orchestrator, with everything build-system-specific
behind `BuildSystemDriver` (`lib/build-systems/`):

```ts
interface BuildSystemDriver {
    readonly id: BuildSystemId;
    readonly manifestFilename: string; // CMakeLists.txt | Cargo.toml | pom.xml
    getUnsupportedReason(compiler: BaseCompiler): string | undefined;
    applyRequestDefaults(compiler: BaseCompiler, parsedRequest: ParsedRequest): void;
    getBuildPath(dirPath: string): string;
    writeProjectFiles(ctx: BuildContext): Promise<{inputFilename: string}>;
    prepareBuildDirectory(ctx: BuildContext): Promise<void>;
    getBuildPlan(ctx: BuildContext): Promise<BuildPlan>; // ordered steps + the effective compiler flags
    getArtifactFilename(ctx: BuildContext): string;
    postProcessArtifact(ctx: BuildContext, result: CompilationResult, artifact: string): Promise<CompilationResult>;
}
```

The orchestrator keeps everything that is not build-system-specific: temp directory management, cache load/store, the
`env.enqueue` compilation queue, remote-execution triple guessing, `afterCompilation`, and cleanup. The two hardcoded
build steps became a loop over `BuildPlan.steps`, each carrying its own name, executable, arguments, exec parameters
and failure placeholder.

The per-compiler hooks (`getExtraCMakeArgs()`, `getCMakeExtToolchainParam()`, `getCmakeBaseEnv()`,
`createCmakeExecParams()`) deliberately keep their CMake-specific names and are now called *by* `CMakeBuildSystem`
rather than by the orchestrator — they are CMake concepts, and a Cargo driver will want different ones rather than a
generic hook that means something different per build system. `win32-vc`, `llvm-mos`, `cc65` and `beebasm` are
untouched.

Two things generalised in passing: `writeAllFilesCMake()` became `writeProjectFiles(dirPath, manifestFilename, …)`,
and `getCmakeCacheKey()` became `getBuildProjectCacheKey(buildSystem, …)`, which sets `api` from the driver id — still
`'cmake'` for CMake, so cached builds survive.

Not yet done, deferred to the phase they are needed by: the execution shape hook (Maven needs `java -jar`; phase 4),
and collapsing `handleCmake` and the SQS `isCMake` branch into a driver lookup (needs the phase 0 wire changes).

`test/build-systems-tests.ts` locks the emitted build plan — step order, argument composition, the shared environment
between the configure and build steps, and the `-GNinja` and toolchain arguments.

### Phase 2 — frontend: an enum, not a boolean

- `MultifileServiceState.isCMakeProject: boolean` becomes `buildSystem: BuildSystemId | 'none'`, migrated on read in
  both `MultifileService` and `ClientStateTree`. Write *both* fields for a transition period so old shortlinks and
  mixed deployments agree, then drop the boolean and regenerate `test/state/*.json`.
- The tree pane's toggle becomes a build-system dropdown filtered by the current language's compatible set. The
  argument input's label, placeholder and default come from the descriptor, so `-DCMAKE_BUILD_TYPE=Debug` stops being a
  global default in `static/components.ts`.
- Deduplicate the pending/next request machinery shared by the compiler and executor panes.
- Replace the `wasCmake` heuristic (sniffing `buildsteps` for a step named `cmake`) with a `buildSystem` field echoed in
  the compilation result.

### Phase 3 — Cargo

Add a `cargo` pseudo-language (manifest `Cargo.toml`, TOML highlighting — a Monaco mode has to be written), force-exposed
like `cmake`. `CargoBuildSystem` is compatible with `rust`, runs `cargo build --message-format=json --offline`,
discovers the artifact from the `compiler-artifact` JSON records, and produces assembly via
`cargo rustc -- --emit asm` through the existing Rust asm parser. Compiler injection is `RUSTC`/`CARGO_HOME` rather than
`CC`/`CXX`. Needs a `cargo=` property.

**Open question to settle before starting:** build nodes have no network access during compilation. v1 is either "no
external crates" or a pre-seeded vendored registry, and the latter is infrastructure work.

### Phase 4 — Maven (then Gradle)

`pom.xml` (XML highlighting), compatible with `java` and `kotlin`. Runs `mvn -o package`, takes the artifact from
`target/*.jar`, post-processes with a jar-aware variant of the existing Java bytecode dump, and executes with
`java -jar`. Same offline-dependency problem as Cargo, but more acute: it needs a pre-seeded `~/.m2` or a local mirror.
Gradle is a cheap follow-on once this shape exists.

## Related issues

Requests that motivate making this pluggable:

- [#3388](https://github.com/compiler-explorer/compiler-explorer/issues/3388) — Maven (or Gradle) dependencies for
  Java. The clearest ask for phase 4.
- [#3919](https://github.com/compiler-explorer/compiler-explorer/issues/3919) — "Can Compiler Explorer support
  language-specific package managers?", asking for Fortran's `fpm`.
- [#7380](https://github.com/compiler-explorer/compiler-explorer/issues/7380) — an `fpm` project template for Fortran,
  explicitly modelled on the existing CMake template.
- [#8988](https://github.com/compiler-explorer/compiler-explorer/issues/8988) — a Rust IDE-mode example; the reporter
  could not get multiple crates working, and fell back to CMake's experimental Rust support.
- [#3763](https://github.com/compiler-explorer/compiler-explorer/issues/3763) — Rust crate support, currently served by
  pre-built Conan binaries rather than Cargo.
- [#5534](https://github.com/compiler-explorer/compiler-explorer/issues/5534) and
  [#2598](https://github.com/compiler-explorer/compiler-explorer/issues/2598) — crate features and `build-std`, both
  things a real Cargo driver would get for free.

Existing CMake bugs that the refactor touches, and should at least not make harder to fix:

- [#5051](https://github.com/compiler-explorer/compiler-explorer/issues/5051) — toolchain switching is ignored under
  CMake because `LD`/`AS`/`AR` come from the compiler's configured toolchain. There is already a `TODO(#5051)` in
  `getCmakeBaseEnv()`; the environment computation moves into the driver, which is where the fix belongs.
- [#2897](https://github.com/compiler-explorer/compiler-explorer/issues/2897) — forgetting `-DCMAKE_BUILD_TYPE` loses
  `-g` and the library filter then hides everything. Phase 2 moves the default arguments into the per-build-system
  descriptor, which is where a real fix would live.
- [#7106](https://github.com/compiler-explorer/compiler-explorer/issues/7106) — CFG is empty in tree mode.
- [#6380](https://github.com/compiler-explorer/compiler-explorer/issues/6380) — shortlink loading breaks when a plain
  compiler and a CMake compiler coexist; relevant to the phase 2 state migration.
- [#6140](https://github.com/compiler-explorer/compiler-explorer/issues/6140) (time-trace),
  [#6550](https://github.com/compiler-explorer/compiler-explorer/issues/6550) (missing ICX asm comments),
  [#4909](https://github.com/compiler-explorer/compiler-explorer/issues/4909) (demangling and source annotation with
  modules) — all cases where the CMake path diverges from the single-file path.
- [#6742](https://github.com/compiler-explorer/compiler-explorer/issues/6742) and
  [#7969](https://github.com/compiler-explorer/compiler-explorer/issues/7969) — MSVC-specific CMake failures, which is
  why `win32-vc`'s `getExtraCMakeArgs()` override must keep working unchanged through phase 1.

## Things not to break

- `/api/compiler/:id/cmake` and `compilerOptions.cmakeArgs` are documented public API. Alias them, never remove them.
- Keep the `'cmake'` cache-key literal, or every cached CMake build is invalidated.
- The SQS message producer lives in another repository; accept both the old and new flags across deploys.
- Old shortlinks must keep rendering forever. Update `test/state/*.json` deliberately, not incidentally.
