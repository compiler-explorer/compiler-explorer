# Build systems

Compiler Explorer's IDE ("tree") mode can hand a whole project to a build system rather than invoking a compiler on a
single source file. CMake, Cargo, Maven and Make are supported, through a driver per build system.

This document describes how that works, and the incremental plan for turning it into a pluggable mechanism so that
Cargo (Rust), Maven/Gradle (Java, Kotlin) and others can be added without another round of special-casing. All the
phases below have landed.

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

- `getCmakeBaseEnv()` sets `CXX`/`CC` (C++), `FC` (Fortran), `CUDACXX` (CUDA), `AS` (assembly), `RUSTC` (Rust), `CC` otherwise.
- `getCompilerEnvironmentVariables()` sets `CXXFLAGS`/`FFLAGS`/`CUDAFLAGS`/`CFLAGS`/`RUSTFLAGS`.
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

   What that default is called is `defaultArtifactName` on the descriptor, so the frontend can show it as the output
   file input's placeholder and the drivers cannot drift from it. It is `output.s` everywhere a build produces a
   binary — the name a plain compilation already uses, so a project only has to learn one — and `output.jar` for
   Maven, which builds a jar. CMake asks the compiler instead of reading it, because a compiler may override the name:
   the Windows ones want a `.exe`.
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

Features of your own package work as normal — declare them in `[features]` and pass `--features`,
`--all-features` or `--no-default-features` in the Cargo arguments box. A library's features cannot be chosen,
because its rlib arrives prebuilt with a fixed set of them; see
[#5534](https://github.com/compiler-explorer/compiler-explorer/issues/5534).

A `[dependencies]` entry therefore always fails, and cargo's own message for it ("no matching package named ...,
location searched: crates.io index") does not hint at what to do instead, so `explainFailure` on the build step adds
that. Making `[dependencies]` genuinely work needs vendored crate *sources* on the build node, which is infrastructure
work and deserves its own phase — see
[#3763](https://github.com/compiler-explorer/compiler-explorer/issues/3763).

### Phase 4 — Maven (done)

`MavenBuildSystem` (`lib/build-systems/maven.ts`) builds `pom.xml` projects for Java and Kotlin, with a `maven` pseudo-language
for the manifest highlighted as XML.

- **JAVA_HOME comes from the selected compiler**, which says which JDK it belongs to: `java_home` for Kotlin,
  `runtime` for Java, and failing both the JDK its own exe sits in. Compilers no JDK can be found for are refused.
- **`maven=` names the mvn to run** — unlike cargo, maven is not part of a toolchain, so it is a property like `cmake`.
- **Maven cannot build anything without its plugins**, and build nodes have no network, so infra's `tools.yaml` primes
  a repository inside the maven install by running each plugin against a throwaway project. The build points
  `maven.repo.local` at it; it is only read, so every compilation shares the one copy. `package` is the default goal,
  left alone if the user names their own.
- **The bytecode is javap over `target/classes`**, reusing the Java compiler's own handling — which is why the driver
  sets `filters.binary`, since that is how Java signals "run javap" rather than anything about native binaries.
- **Execution needs a JVM started on the classes**, which is what the `prepareExecution` hook is for, along with the
  same JVM flags the Java compiler uses to fit inside the sandbox's thread and memory limits. The main class is found
  by looking for the `main` descriptor in each class file's constant pool.
- **`JavaCompiler.readdir` is now recursive**, because anything with a package puts its classes in a matching
  directory tree, which a build system does by default. Clojure had already overridden it for the same reason.

A `<dependencies>` entry cannot resolve, and `install` fails because it writes to the shared read-only repository.
Both get an explanation rather than a bare Maven error. The explanation is given the step's whole output, not just
its stderr, because maven says everything on stdout.

### Phase 5 — Kotlin under Maven (done)

The same driver, with `kotlin` added to the descriptor's languages. Three things had to give:

- **A Kotlin compiler is not part of a JDK**, living in its own `kotlin-jvm-x.y.z`, so JAVA_HOME had to come from what
  the compiler declares rather than from where its exe sits.
- **`kotlin-maven-plugin` compiles with a compiler it resolves from the repository**, not with anything installed on
  the machine, which left to itself would make the compiler picker decorative — the pom would decide. Two things fix
  that, and together they mean the compiler you select is the one that runs:
  - **The version is told to the plugin**: `-Dkotlin.version=<selected semver>`, which a pom following the convention
    of naming its Kotlin once picks up for the plugin and the standard library alike. It goes before the user's own
    arguments, so a project that insists on a version still gets it.
  - **The jars come from the selected installation.** A Kotlin installation's `lib/*.jar` are the very Maven
    artifacts, byte for byte — `kotlin-compiler.jar` is `org.jetbrains.kotlin:kotlin-compiler` down to the sha1. So
    the driver symlinks them into a repository of the build's own and puts it in front, using
    `maven.repo.local.tail`, the chained local repository Maven Resolver 1.9 (Maven 3.9) provides. Three links,
    nearest first: the build's own, then `kotlin-jvm-<version>-maven` — installed beside the compiler by infra's
    `tools/kotlin-maven`, holding only what the installation does not carry — then the shared repository maven was
    installed with, which holds the Java plugins and never changes when a Kotlin is added. A Kotlin whose repository
    is not installed is refused up front, naming the versions that are.
  - From **Kotlin 2.2** the plugin drives the compiler through the build tools API and asks for
    `kotlin-compiler-embeddable`, a repackaging with everything relocated inside it that no distribution ships. That
    one is bundled, at 54 MiB a version. It is still the Kotlin release that was selected, only JetBrains' embedding
    build of it rather than the command-line one.
- **Running a Kotlin program needs its standard library beside the classes.** Maven leaves dependencies in the shared
  repository, which the execution sandbox cannot see, so `prepareExecution` has maven copy them into the project
  first — named in full, since resolving the `dependency:` prefix would need metadata from Maven Central. It also
  needs a bigger stack than the Java compiler's `-Xss136K`: reaching Kotlin's collections overflows it during the
  nested class loading before `main` is entered.

### Phase 6 — Make (done)

`MakeBuildSystem` (`lib/build-systems/make.ts`) runs one `make` against a `Makefile`, and is the first build system
offered for **every** language: `compatibleLanguageIds: 'all'`, since a Makefile says for itself what to run rather
than being tied to a toolchain. One `make=` in the properties names the binary, as `cmake=` does.

- **It is handed the environment CMake is handed**, by the same `createCmakeExecParams`: `CXX`/`CC`/`FC`/`CUDACXX`/
  `AS`/`RUSTC` from the selected compiler, the matching `CXXFLAGS`/`CFLAGS`/`FFLAGS`/`CUDAFLAGS`/`RUSTFLAGS` carrying
  the user's options and libraries, and `LDFLAGS`. So a recipe reading `$(CXX) $(CXXFLAGS) -o output main.cpp`
  compiles with what was selected in the UI, which is the whole point of offering it.
- **`NVCC` as well, when the compiler really is nvcc.** CMake has no need of it, but a CUDA Makefile conventionally
  says `$(NVCC)`, and make has no built-in for it, so it would otherwise expand to nothing and the recipe would run
  without a compiler. Guarded on `compilerType === 'nvcc'`: clang compiles CUDA too, and naming it NVCC would be a
  lie the Makefile cannot see through. Note the converse trap, which is CMake's too: for CUDA, `$(CXX)` is make's
  own built-in `g++` rather than anything selected here, since CE sets `CUDACXX` for that language.
- **Nothing is added to the command line.** A bare `make` is the default target; targets, `-j`, variable overrides
  are the user's to pass.
- **The artifact is `output` unless the project says otherwise**, because only the Makefile knows what it built.
  When nothing is there, the failure says so and points at the output file box rather than letting the disassembler
  fail on a missing file.
- **Makefiles are edited with tabs**, whatever the indentation setting says, since a recipe line that begins with
  spaces gets only "missing separator" from make (`static/panes/editor.ts`).

## Things not to break

- `/api/compiler/:id/cmake` and `compilerOptions.cmakeArgs` are documented public API. Alias them, never remove them.
- Keep the `'cmake'` cache-key literal, or every cached CMake build is invalidated.
- The SQS message producer lives in another repository; accept both the old and new flags across deploys.
- Old shortlinks must keep rendering forever. Update `test/state/*.json` deliberately, not incidentally.
