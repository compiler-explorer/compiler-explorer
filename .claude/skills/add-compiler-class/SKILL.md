---
name: add-compiler-class
description: Write a new BaseCompiler subclass (compilerType) or another pluggable class (demangler, objdumper, formatter, tool, build env, storage, shortener, asm docs) and its unit test. Use when an existing compilerType can't handle a compiler, or when adding a new tool/formatter/demangler class.
---

# Add a compiler class or other pluggable class

Configuring a compiler that an existing class already handles is a properties-only job: see
`docs/AddingACompiler.md` and the compiler-config agent. This skill covers the case that doc calls out of scope:
writing the class.

## Registration (same for every pluggable family)

1. Create the class with `static get key()` returning the id used in properties (a string or an array of aliases).
2. Export it from the family's `_all.ts`, in alphabetical order. Each `index.ts` builds the lookup with
   `makeKeyedTypeGetter` (`lib/keyed-type.ts`).
3. Select it from properties. The key for each family:

| Directory | Property |
|---|---|
| `lib/compilers` | `compilerType` |
| `lib/demangler` | `demanglerType` |
| `lib/objdumper` | `objdumperType` |
| `lib/formatters` | `formatter.X.type` (also see `docs/AddingAFormatter.md`) |
| `lib/tooling` | `tools.X.class` (also see `docs/AddingATool.md`) |
| `lib/buildenvsetup` | `buildenvsetup` |
| `lib/external-parsers` | `externalparser` |
| `lib/storage` | `storageSolution` |
| `lib/shortener` | `urlShortenService` |
| `lib/asm-docs` | `instructionSet` |

Mistakes show up at startup: a missing or duplicate key throws `<family> KeyedType configuration error`, and a
key that isn't exported throws `No <family> named '<key>' found`.

## Compiler classes

- Extend `BaseCompiler` or the closest existing family (`GCCCompiler`, `ClangCompiler`, ...). Read the sibling
  classes for that family first; most new compilers are small variations of one.
- Override hooks, not `compile`/`doCompilation`. The most-overridden hooks, roughly in order: `optionsForFilter`,
  `getArgumentParserClass`, `runCompiler`, `getOutputFilename`, `processAsm`, `orderArguments`,
  `filterUserOptions`, `getDefaultExecOptions`, `objdump`. To use a different asm parser, assign `this.asm` in the
  constructor (see the change-asm-filtering skill).
- View support flags (`supportsXView`) are set in the constructor. Adding a new view is the add-output-pane skill.

## Tests

- Construct the class with `makeFakeCompilerInfo({exe: '/dev/null', lang: '<lang>', ...})` and
  `makeCompilationEnvironment({languages})` from `test/utils.ts`, then mock process execution with
  `vi.spyOn(compiler, 'exec')` and test the hooks you overrode (`optionsForFilter`, `postProcess`, ...).
  Examples: `test/golang-tests.ts`, `test/mach-tests.ts`, `test/compilers/*-tests.ts`.
- Name the file `*-tests.ts` under `test/`. Every `.ts` file there is collected as a test, except
  `_*.ts` and `utils.ts`.
- Finish with a real compile in `make dev` against an actual binary, configured in a `.local.properties` file.
