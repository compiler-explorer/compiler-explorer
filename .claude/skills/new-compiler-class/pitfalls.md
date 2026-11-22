# Mistakes existing classes make

Each of these is in more than one class in `lib/compilers/`, so copying an existing class is how they spread.
Check a new class, or one under review, against this list.

## Concurrency

There is one instance per configured compiler, and all of that compiler's compilations run through it at once.

- **State on `this` set during a compilation.** For example a hook stores the library paths, the output extension or
  an option it found, and a later hook reads it. Two compilations overwrite each other's value. Work it out again from
  the hook's arguments each time, or pass it along.
- **Changing `this.compiler` during a compilation.** For example swapping `this.compiler.exe` to an interpreter
  around an `await`. Every other compilation in flight runs the wrong program. Pass the program to run as an argument
  (`this.exec(otherExe, ...)`, `runExecutable(otherExe, ...)`).

## Arguments

- **Changing filters in `optionsForFilter`, `processAsm` or `postProcess`** (`filters.binary = true`).
  Force or drop filters in `fixFiltersForCompiler`, and see the `compiler-class-cache-key` skill.
- **Changing the arrays you are given.** `options`, `userOptions` and `key.options` are reused by the caller
  (`push`, `splice`, `shift` and `pop` all change them). Build a new array.
- **Finding the user's options by position**, e.g. "everything before the first `.java`". The input file can appear
  more than once, or not at all. Use the `userOptions` parameter, or `filterUserOptions`.
- **Dropping the user's options or the configured `options`.** An override of `prepareArguments` or `runCompiler`
  that rebuilds the command must keep both, plus library and backend options. If an option can't work, remove just
  that one in `filterUserOptions`.
- **One argument with a space in it** (`'-Fc ' + file`), **literal quotes** (`'-X="a"'`), or **empty strings**
  (`['']`). Arguments aren't passed through a shell: each one is one array element.
- **Relative output paths** (`-o output.txt`) that only work because of the cwd. Use the `outputFilename` you are given.
- **Paths interpolated into code** (a script passed with `-eval`) without escaping.

## Running programs

- **Calling `exec.execute` or `child_process` directly.** Use `this.exec()`, the hook that Wine and WSL subclasses
  override, or `this.execCompilerCached()` for results that don't change between runs.
- **Exec options built from scratch** (`{maxOutput: ...}`). Without `getDefaultExecOptions()` there is no timeout and
  no compiler wrapper (the sandbox in production). Start from `this.getDefaultExecOptions()` and change what you need.
- **Replacing `env`** (`env = {PATH: ...}`, or a clone of `process.env`). That drops what CE set up. Add to
  `execOptions.env`.
- **Setting `customCwd` to the input's directory.** `runCompiler` already does that.
- **Hard-coded install paths** (`/opt/compiler-explorer/...`, a home directory), or a layout guessed from the exe
  path. Make it a property, with a default only where the layout is the compiler's own.
- **Behaviour chosen by compiler id or by a string in the exe path.** Make it a property, or compare `semver`.
- **Running a probe on every compilation** (`--print-prefix` and the like). Do it once in `initialise()`.
- **Synchronous file I/O** in a hook. Use `fs/promises`.

## Startup

- **No `getArgumentParserClass()` override for a compiler that isn't GCC-like.** The fallback is `GCCParser`, which
  runs `-fsyntax-only --help` six ways on every start and feeds the answers to the option tooltips. Return `BaseParser`
  unless a parser for the compiler's own help exists.
- **Borrowing another language's parser** (a Clang or Pascal parser for an unrelated compiler).

## Panes and output

- **Claiming a pane that doesn't work.** `supportsIntel` without `optionsForFilter` honouring `filters.intel`,
  `supportsIrView` or `optPipeline` with an empty argument, `isCfgCompiler()` for output the CFG parser can't read.
- **Reusing a toggle for something else** (the Intel switch meaning "show the VM code").
- **Doing expensive work nobody asked for**, e.g. always building IR, or running the build twice.
- **Throwing when the output is unexpected.** Show it as it is, or return a result with a non-zero `code` and a
  message, so the user sees what happened.

## Errors and code hygiene

- **Empty `catch {}`.** Log it with `logger`, or comment why the error doesn't matter.
- **`console.log`, `console.assert`, throwing strings.** Use `logger`, `assert` and `Error`.
- **Overrides that only call `super`**, and empty library overrides copied without checking what the compiler
  accepts. Delete them.
- **Copying a whole base method** to change one line (`prepareArguments`, `getVersion`, `postProcess`). Override a
  smaller hook, or call `super` and change the result.
- **Two near-identical classes.** Make one class configurable by a property, or put the shared part in a base class.
