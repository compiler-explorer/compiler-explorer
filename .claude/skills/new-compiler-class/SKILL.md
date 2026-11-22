---
name: new-compiler-class
description: How to write a new compiler class (compilerType) in lib/compilers/ for Compiler Explorer, from investigating what the compiler can do through registering, testing and trying it locally. Use when someone wants to support a compiler or language whose command line or output doesn't fit an existing compilerType, asks to "add a compiler class", or is reviewing a PR that adds one.
---

# Writing a new compiler class

A compiler class turns a compilation request into a command line, runs it, and turns the output into what the
panes show. `BaseCompiler` (`lib/base-compiler.ts`) does all of that for a GCC-like driver. A new class only
overrides the parts where the compiler differs.

Reference files in this skill:
- `hooks.md`: which hook to override for each need, with an existing class that does it.
- `pitfalls.md`: the mistakes existing classes make. Check the new class against it before the PR, and use it when
  reviewing one.

## 1. Find out what the compiler can do

Every later step depends on this. Run the compiler by hand on a small example in an empty directory, and write down
the exact commands that work. They become `optionsForFilter` and the tests.

**Input**
- Which language does it compile, with which file extension? Is the language in `lib/languages.ts`? If not, also
  follow `docs/AddingALanguage.md`.
- Does it need a particular file or module name, a `main`, or a project or manifest file next to the source?

**Output**
- What can it produce: assembly, an object file, an executable, IR, bytecode, source in another language? Each one
  decides which panes and filters the class supports.
- Which flags or subcommand produce each one, and Intel vs AT&T syntax if it has both?
- Where does the output go: `-o`, next to the input, a fixed name, a directory, several files, or stdout?
  List the directory after each run.
- Does it write other files (caches, dumps, debug info)? Does the output contain the absolute temp path?
- Can it emit line information that maps the output back to the source?
- Is the output large? A big runtime linked in or a full listing can exceed the output limit.

**Running it**
- Does it need environment variables, a writable `HOME` or cache dir, or other tools (assembler, linker,
  disassembler, a host C compiler, a runtime)? Run it with a clean environment from another directory to find out.
  Each tool it needs becomes a property.
- Does it need the network (package registry, downloads)? The sandbox has none.
- Can two runs share a directory without clashing?
- Is it a script that needs an interpreter?

**Options**
- Which user options would break the class: `-o`, `--help`, run modes, output directories?
- Does it read arguments starting with `/` as options, or paths with spaces badly?
- Which flags select a target or a language standard?
- Do flags or output change between versions?

**Everything else**
- How does it report its version?
- What does its help output look like? GCC's format, its own, or none?
- How does it report errors? Compile code with an error in it: is it `file:line:col`, and are columns zero-based or
  counted in bytes?
- How does it take include paths and libraries, if at all?
- Can its output run here, natively, under an emulator, or through an interpreter or JIT?

## 2. Check that a class is needed

Configuration alone covers a lot, see `docs/AddingACompiler.md`. Try these first, in a `.local.properties` file:

- An existing `compilerType`: `gcc`, `clang` and most language keys already handle their compilers' variants.
- `options`, `intelAsm`, `versionFlag`, `versionRe`, `explicitVersion`, `supportsBinary`, `supportsExecute`,
  `demangler`, `objdumper`, `executionWrapper`.

Write a class only when what you found in step 1 can't be expressed that way. A class that only sets a `key` is
a sign that config would have done.

## 3. Pick the base class

| The compiler is | Extend |
|---|---|
| A GCC or Clang build with a few quirks (a fork, a vendor toolchain, a cross compiler) | `GCCCompiler` / `ClangCompiler` |
| A variant of a language CE already supports (another Rust, .NET or Java backend) | That language's class |
| Anything else | `BaseCompiler` |

`BaseCompiler`'s defaults assume a GCC-style driver. A class for a compiler that isn't one almost always needs:
- `optionsForFilter`: the default is `-g -o <out>` plus `-S` or `-c`, Intel and verbose-asm flags.
- `getArgumentParserClass`: the default falls back to `GCCParser`, which runs `-fsyntax-only --help` six ways on
  every start. Return `BaseParser` unless the compiler has a help format worth parsing.
- The library hooks, only if it rejects `-I`, `-L` or `-l`: `getIncludeArguments()`, `getLibLinkInfo()`.

## 4. Override only what differs

Find each thing step 1 turned up in `hooks.md`. Read the example class it names, and write the narrowest override
that does the job. Rules that apply to every override:

- One instance serves all concurrent compilations of a compiler. Don't keep per-compilation state on `this`, and
  don't change `this.compiler` during a compilation: work it out again from the hook's arguments.
- Don't change the `options`, `userOptions` or `filters` you are given, except in the hooks meant for it.
  Force or drop filters in `fixFiltersForCompiler()`.
- Run programs with `this.exec()`, starting from `this.getDefaultExecOptions()`. Add to `env`, never replace it.
- Pass paths through `this.filename()` before they go on the command line.
- Settings that vary per install are properties, not hard-coded paths. Read per-compiler ones with
  ``compilerProps(`compiler.${this.compiler.id}.x`)``, falling back to the group and language.
- Don't override `compile()` or `getCacheKey()`, see the `compiler-class-cache-key` skill.

## 5. Register it

1. `static get key()` returns the name used as `compilerType` in the properties files.
2. Export the class from `lib/compilers/_all.ts`, in alphabetical order.
3. Set `compilerType=<key>` on the compiler or its group in `etc/config/<lang>.*.properties`, and add a property
   for each tool or setting the class reads.
4. Add the class file to the language's entry in `.github/labeler.yml`.

A new language needs more than this; follow `docs/AddingALanguage.md` for the rest.

## 6. Test it

Unit-test what the class decides, without running the compiler: start from
`.claude/skills/new-compiler-class/templates/compiler-tests.ts`, saved as `test/<key>-tests.ts`. Cover every branch
of `optionsForFilter` (asm, binary, binary object, any user option it reacts to), expecting the commands that worked
by hand in step 1. Also cover `filterUserOptions` and every other override that computes something. Run it with
`npm run test -- --run <key>-tests.ts`.

Then run it for real:

1. Set `delayCleanupTemp=true` in `etc/config/compiler-explorer.local.properties` so temp dirs stay on disk after a compile.
2. Add the compiler to `etc/config/<lang>.local.properties`.
3. `make dev EXTRA_ARGS='--debug --language <lang>'`, and check it's in `http://localhost:10240/api/compilers/<lang>`.
   Startup shouldn't log `--help` probes the compiler doesn't understand.
4. In the browser, try every mode the class claims: asm, binary, binary object, execution, Intel syntax, and each
   pane it enables. Then try user options (including the ones step 1 found would break it), and code that fails to
   compile: the errors should show in the editor.
   Look in the kept temp dir when the output is wrong.
5. Remove `delayCleanupTemp=true` afterwards.

## Before the PR

- Go through `pitfalls.md` against the new class.
- `npm run ts-check`, `npm run lint`, `npm run test`.
- For the public site the compiler also needs an installer in the infra repo, see `docs/AddingACompiler.md`.
