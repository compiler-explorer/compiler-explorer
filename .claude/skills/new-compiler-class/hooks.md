# Which hook to override

Override the narrowest hook that does the job, and call `super` when you only need to add to it. Each row names a class
that does it reasonably well. Read that class before writing your own, but don't copy it wholesale: most classes carry
some of the flaws in `pitfalls.md`.

## The command line

| To change | Override | Example |
|---|---|---|
| The flags for the requested output, including a subcommand (`build`, `compile`) | `optionsForFilter(filters, outputFilename, userOptions)` | `hylo.ts`, `c3c.ts`, `codon.ts` |
| Map the base flags instead of replacing them | call `super.optionsForFilter()` and edit the result | `mojo.ts` |
| User options that would break the command (`-o`, `--help`, run modes, output dirs) | `filterUserOptions(userOptions)` | `zig.ts`, `java.ts` |
| Where input, user options and library flags go (a `--` separator, input before options) | `orderArguments(...)` | `c2rust.ts`, `elixir.ts` |
| Flags for a target or language standard picked in the UI | `getTargetFlags()`, `getStdverFlags()`, `populatePossibleOverrides()` | `tablegen.ts` |
| The whole argument assembly | `prepareArguments()` | Last resort: an override must still add the configured `options`, libraries and backend options, which most copies forget. |
| Library flags | `getIncludeArguments()`, `getLibLinkInfo()`, `getSharedLibraryPathsAsArguments()`, `getSharedLibraryLinks()` | Return `[]` only for flags the compiler rejects. `getLibLinkInfo` drops all link flags in one place. |

## Files

| To change | Override | Example |
|---|---|---|
| The input file name (a required class or module name) | set `this.compileFilename` in the constructor | |
| The output name, when the compiler picks it | set `this.outputFilebase` in the constructor, or `getOutputFilename()` | `tic2000.ts`, `vyper.ts` |
| The executable or IR file name | `getExecutableFilename()`, `getIrOutputFilename()` | `ispc.ts`, `c3c.ts` |
| Extra files the compiler needs (project file, manifest, stub driver) | `writeAllFiles()`, or write them in `runCompiler()` | `mach.ts`, `gleam.ts` |
| Source the compiler can't take as-is (needs a `main`) | `preProcess(source, filters)` | `pony.ts` |

`getOutputFilename` and `getExecutableFilename` are also called without a `key`: check it with `isCacheKey()` before
reading from it. `getExecutableFilename` defaults to `getOutputFilename`, so override it only when they differ.

## Running the compiler

| To change | Override | Example |
|---|---|---|
| Environment that every run needs (PATH, cache dirs, tool roots) | `getDefaultExecOptions()`: call `super`, then add to `env` | `tinygo.ts`, `qnx.ts` |
| Directories to add to PATH | `getExtraPaths()` | `win32-mingw-gcc.ts` |
| Environment that differs per compilation (a cache dir inside the temp dir) | `runCompiler()`: add to `execOptions.env`, then `super.runCompiler()` | `v.ts` |
| A wrapper script around the compiler | a script in `etc/scripts/` (`resolvePathFromAppRoot`), overridable by a language property, passed in `optionsForFilter`. Don't use the base's `compiler-wrapper`/`this.compilerWrapper`: that is the production sandbox wrapper. | `triton.ts`, `python.ts` |
| Several tools in a row (compile, then disassemble or translate) | `runCompiler()`: run each step with `this.exec`, stop at the first failure, merge stdout/stderr | `glsl.ts`, `spirv.ts` |
| Another compiler CE already has configured | `this.env.findCompiler(lang, id)` | `r8.ts`, `wyrm.ts` |

`runCompiler` already defaults the cwd to the input's directory. A `runCompiler` that doesn't call `super` must still
return `languageId`, `instructionSet` and `inputFilename`: build the result with `transformToCompilationResult()`.

## Output

| To change | Override | Example |
|---|---|---|
| Output on stdout instead of a file | write stdout to the output file in `runCompiler()` | `nix.ts`, `beebasm.ts` |
| Output spread over several files | `checkOutputFileAndDoPostProcess()` | `solidity-zksync.ts` |
| Something other than assembly (IR, bytecode, another language) | `getCompilerResultLanguageId()`, plus a parser: `AsmParserCpp` for source text, rather than writing a no-op parser | `vala.ts`, `cppfront.ts`, `codon.ts` |
| Parsing the output | `this.asm = new SomeParser()` in the constructor, or `processAsm()` | `pythran.ts`, `vyper.ts` |
| Disassembling a binary with a tool other than objdump | `objdump()`, `getObjdumpInputFilename()`, `supportsObjdump()` | `java.ts`, `racket.ts` |
| Errors CE doesn't recognise as `file:line:col` | fix them up after `runCompiler`, or map columns | `lean.ts`, `glsl.ts` |
| Turning off filters that make no sense for this compiler | `disabledFilters` in the info passed to `super()` | `java.ts`, `mlir.ts` |
| Forcing a filter (always a binary, never a binary) | `fixFiltersForCompiler()` | `tinygo.ts`, `micropython.ts` |

## Extra panes

Only enable a pane once it works, and only with a real argument behind it.

| Pane | Set / override | Example |
|---|---|---|
| IR | `supportsIrView`, `irArg`, and `getIrOutputFilename()` or `generateIR()` | `c3c.ts`, `flang.ts` |
| Opt pipeline | `optPipeline`, `generateOptPipeline()` | `llc.ts`, `numba.ts` |
| Device code | `supportsDeviceAsmView`, `extractDeviceCode()` | `nvcc.ts`, `triton.ts` |
| AST | `generateAST()`, `couldSupportASTDump()` | `ldc.ts`, `python.ts` |
| CFG | `isCfgCompiler()` | Only for output the CFG parser understands. |
| Intel syntax | `supportsIntel`, `intelAsm` | `optionsForFilter` must honour `filters.intel`. |

## Execution

| To change | Override | Example |
|---|---|---|
| Run source with an interpreter or JIT instead of a built binary | `handleInterpreting()` | `typescript-native.ts`, `java.ts` |
| Run the binary differently (a runtime, an emulator) | `runExecutable()`, or `executionWrapper` in config | `ldc.ts`, `gleam.ts` |
| Build the executable differently from the asm compile | `buildExecutable()` / `buildExecutableInFolder()` | `gleam.ts`, `mach.ts` |
| Return a binary to download | `addArtifactToResult()` | `llvm-mos.ts`, `z88dk.ts` |
| Tell the user something about the result | `result.hints` | `beebasm.ts`, `cc65.ts` |

## Startup

| To change | Override | Example |
|---|---|---|
| Option discovery | `getArgumentParserClass()`: `BaseParser` when the help format isn't GCC's, or a parser for the compiler's own | `tablegen.ts`, `zig.ts`, `ylc.ts` |
| Version probe | `versionFlag`, `versionRe` or `explicitVersion` in config. Override `getVersion()` only when those can't do it. | `norcroft.ts` |
| Features that depend on the version | compare `this.compiler.semver` with `Semver` | `c3c.ts`, `zig.ts` |
| Anything probed once per compiler (a runtime path, supported targets) | `initialise()`, then `super.initialise()` | `mach.ts` |
| A per-compiler setting | ``this.compilerProps(`compiler.${this.compiler.id}.name`)`` with group and language fallbacks. A bare `compilerProps('name')` only reads the language-level key. | `lua.ts`, `tinygo.ts` |

## CMake

Only when the compiler should work in CMake mode: `getExtraCMakeArgs()` for arguments CMake needs, and
`getCmakeBaseEnv()` for its environment (`cc65.ts`, `clangcl.ts`).
