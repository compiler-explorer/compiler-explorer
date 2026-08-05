# Build systems

Compiler Explorer's IDE ("tree") mode can hand a whole project to a build system rather than invoking a compiler on a
single source file. CMake and Cargo are supported, through a driver per build system.

This document describes how that works, and the incremental plan for turning it into a pluggable mechanism so that
Cargo (Rust), Maven/Gradle (Java, Kotlin) and others can be added without another round of special-casing. Phases 0-3
below have landed; phase 4 has not.

## How it works

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
   from `ceProps('useninja')`, plus the toolchain param and the user's build system arguments), then
   `cmake --build .`.
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
| HTTP route         | `POST /api/compiler/:id/build/:buildSystem`, and the original `/cmake` (`lib/handlers/compile.ts`) |
| Sub-server proxying| `compilerInfo.cmakePath` and `buildPath` (`types/compiler.interfaces.ts`, set in `lib/compiler-finder.ts`) |
| Queue worker       | `RemoteCompilationRequest.buildSystem`, or the older `isCMake` boolean (`lib/compilation/sqs-compilation-queue.ts`) — produced outside this repo |
| Cache key          | `CmakeCacheKey`'s `api: 'cmake'` discriminator (`types/compilation/compilation.interfaces.ts`) |
| Stats              | the build system id as the build method (`lib/stats.ts`)                               |
| Metrics            | `ce_project_build_*` labelled by build system, plus the older `ce_cmake_*`, and SQS equivalents of both |
| Config             | `cmake=` and `useninja=` in `etc/config/compiler-explorer.*.properties`                |

### Frontend

- `MultifileService` holds `buildSystem: BuildSystemId | 'none'`, migrated from the older `isCMakeProject` boolean.
  The main source file is whichever manifest the chosen build system declares.
- The tree pane (`static/panes/tree.ts`) has a build system dropdown, filled from `getBuildSystemsForLanguage()`, plus
  the `cmakeArgs` and `customOutputFilename` inputs — all part of `TreeState`.
- The compiler and executor panes each duplicate a parallel request path — `sendBuildCompile()`,
  `pendingBuildRequestSentAt`, `nextBuildRequest`, `onBuildResponse()` — taken when the tree has a build system.
- `CompilerService.submitBuild()` posts to `/build/<buildSystem>`.
- Each manifest language (`cmake`, `cargo`) is a pseudo-language in `lib/languages.ts`, force-exposed to the frontend
  even when no compiler claims it (`lib/handlers/api.ts`, `lib/app/config.ts`) so tree mode can always resolve it.
- Shortlinks persist the choice through `ClientStateTree` (`lib/clientstate.ts`), still writing `isCMakeProject`
  alongside it; `test/state/*.json` are golden fixtures of that format.

## Why CMake didn't generalise

The plumbing was the easy part. The hard part was four assumptions baked into the old `cmake()`, which is what the
driver interface exists to separate:

1. **The compiler is selected via C-style environment variables.** Cargo wants `RUSTC`/`CARGO`; Maven wants a JDK on
   `JAVA_HOME`.
2. **The artifact lives at a conventional path.** Cargo needs `--message-format=json` (or `cargo metadata`) to discover
   `target/debug/<name>`; Maven produces `target/*.jar` named from the POM. Note discovery supplies the *default* only:
   `backendOptions.customOutputFilename` is the user picking which artifact to inspect, and a build with several
   targets — extra bins, libraries, examples — has plenty of others worth looking at. A driver that discovers its
   artifact must still let that override win.
3. **"Output" means disassembling a native binary.** For Cargo the useful output is `cargo rustc -- --emit asm`; for
   Maven it is javap-style bytecode, and there is no native binary at all.
4. **Execution means running the artifact directly.** Maven artifacts need `java -jar`.

So the abstraction has to cover manifest handling, build steps, environment, artifact discovery, post-processing and
execution shape — not just "which binary do we run".

## Plan

Phases 0–2 are refactoring with no user-visible change. Each is independently shippable.

### Phase 0 — protocol groundwork (done)

`shared/build-systems.ts` holds the `BuildSystemId` type and the descriptors both frontend and backend need: display
name, manifest filename, manifest language id, compatible language ids, default arguments, argument-input placeholder.
The backend drivers in `lib/build-systems/` each carry their descriptor.

The wire was widened in back-compatible ways:

- New route `POST /api/compiler/:id/build/:buildSystem`, with `/cmake` kept **permanently** as an alias (it is
  documented public API in `docs/API.md`). Both go through `CompileHandler.handleProjectBuildWith()`.
- `backendOptions.buildSystemArgs`, falling back to `cmakeArgs` — see `getBuildSystemArgs()`. The frontend still sends
  `cmakeArgs`, so cache keys are unchanged until phase 2 switches it over.
- `CmakeCacheKey.api` is set from the driver id, still `'cmake'` for CMake, so existing cache and executable-package
  hashes stay valid.
- `RemoteCompilationRequest.buildSystem` alongside a still-honoured `isCMake` — the producer of those messages lives in
  another repository, so both must work across a rolling deploy.
- `compilerInfo.buildPath` alongside `cmakePath`. Optional, because a sub-server on an older version has neither the
  field nor the route; CMake keeps proxying to `cmakePath` for exactly that reason.
- Stats record the build system id as the build method. The `ce_cmake_*` counters still count CMake for existing
  dashboards, and `ce_project_build_*` / `ce_sqs_project_build_*` count every build system with a `build_system` label.

**Infra dependency:** the ALB has overriding routes for `/api/compiler/*/compile` and `/api/compiler/*/cmake` per
environment. They need `/api/compiler/*/build/*` adding before the frontend starts using the new route —
[compiler-explorer/infra#2269](https://github.com/compiler-explorer/infra/issues/2269).

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

Not yet done, deferred to the phase it is needed by: the execution shape hook, since Maven needs `java -jar`
(phase 4).

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

### Phase 3 — Cargo (done)

`CargoBuildSystem` (`lib/build-systems/cargo.ts`) builds `Cargo.toml` projects for the `rust` language, alongside a
`cargo` pseudo-language for the manifest and a TOML Monaco mode (`static/modes/toml-mode.ts` — Monaco ships `ini`, but
TOML's array-of-tables headers, arrays and inline tables all come out wrong under it, and a Cargo.toml is full of them).

Points worth knowing:

- **cargo comes from the selected compiler's own toolchain**, not a `cargo=` property: it is the sibling of that
  compiler's `rustc`. A 1.80 cargo driving a 1.91 rustc disagree about lockfile and edition features. Rust compilers
  that ship no cargo — gccrs, mrustc, the BPF gcc — are refused by `getUnsupportedReason`, which is why that hook is
  async: it has to stat for the binary.
- **The compiler is injected as `RUSTC`**, with user options going through `RUSTFLAGS`, and `CARGO_HOME` pointed inside
  the sandbox so nothing cargo writes outlives the compilation.
- **Output goes to `--message-format=json-render-diagnostics`**: cargo then renders diagnostics to stderr, which is
  what the user reads, and leaves stdout as artifact records for us. The driver parses those records, then blanks that
  stdout so the JSON never reaches the output pane.
- **cargo names its output after the manifest**, so `finaliseArtifact` copies what it built to the path the rest of the
  compilation was told to expect. `customOutputFilename` picks between artifacts when a project builds several.
- **The paths cargo reports are the ones it saw**: the sandbox bind-mounts the project as `/app`, and without a sandbox
  it is the real temp directory. `utils.maskRootdir` reduces both to a path relative to the project root, which is then
  rebased onto the real one.

**Libraries come from the Libraries pane, not `[dependencies]`.** Compiler Explorer's Rust crates are prebuilt
`.rlib`s fetched from Conan, and `setupBuildEnvironment` unpacks them into the project before the sandboxed build.
cargo cannot resolve them itself — it wants sources from a registry, and there is no network on a build node — so the
driver passes them to rustc directly as `--extern`, through `CARGO_ENCODED_RUSTFLAGS`. `use rand::Rng;` then works
with nothing declared in Cargo.toml.

A `[dependencies]` entry therefore always fails, and cargo's own message for it ("no matching package named ...,
location searched: crates.io index") does not hint at what to do instead, so `explainFailure` on the build step adds
that. Making `[dependencies]` genuinely work needs vendored crate *sources* on the build node, which is infrastructure
work and deserves its own phase — see
[#3763](https://github.com/compiler-explorer/compiler-explorer/issues/3763).

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
