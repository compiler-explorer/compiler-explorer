---
name: tool-info-sources
description: Where a tool in lib/tooling/ (clang-tidy, PVS-Studio, llvm-mca, ...) should get each piece of information it needs, and which objects it must not modify. Use when writing a new tool, changing an existing one, or reviewing tool code, and when anything reads compiler settings through a compilation cache key.
---

# Where tools get their information

A tool runs after (or instead of) a compilation and receives three things: its own configuration, a
`CompilationInfo` describing the compilation, and the user's arguments for the tool. Each piece of information has
one right place to come from. Getting it from the wrong place either silently breaks the tool or changes the
compilation cache key.

## Look it up in this order

| You need | Get it from | Notes |
|---|---|---|
| The tool's exe, its configured options, exclusions, language | `this.tool` (`ToolInfo`) | Set per tool in the `.properties` files. |
| Site-wide or per-compiler properties | `this.env.ceProps` / `this.env.compilerProps` | Only for settings that are not already on `this.tool` or the compiler. |
| The compiler's own settings | `compilationInfo.compiler` | Typed `ToolCompilerInfo`: only the fields listed below. |
| What this compilation used: the user's compiler options, libraries, filters | `compilationInfo.options`, `.libraries`, `.filters` | Fine to read, never modify, see below. |
| The full command line the compilation actually ran | `compilationInfo.compilationOptions` | The compiler's configured options, the user's, `-o`, the input file. |
| Where the compilation's files are | `compilationInfo.inputFilename`, `.dirPath`, `.outputFilename`, `.executableFilename` | |
| How to run things like the compilation did | `compilationInfo.execOptions`, `.preparedLdPaths` | |
| What the user typed for the tool | the `args` and `stdin` parameters of `runTool` | Read only, see below. |

## The compiler settings a tool can see

`compilationInfo.compiler` is the compiler's `CompilerInfo`, but typed as `ToolCompilerInfo`
(`types/compilation/compilation.interfaces.ts`), which allows only what tools are known to need:

| Field | Used for |
|---|---|
| `id` | identifying the compiler |
| `exe` | running the compiler, or finding a tool installed next to it |
| `options` | the compiler's configured options, to reproduce its command line |
| `lang` | the language being compiled |
| `compilerType` | telling compiler families apart (clang, gcc, ...) |
| `group` | a compiler's group name; prefer `compilerType` for telling families apart |
| `instructionSet` | the target, for tools that analyse the output |
| `includePath`, `libPath`, `envVars` | MSVC's environment |

If a tool needs another field, add it to `ToolCompilerInfo` in the same change, so the list stays a deliberate record
of what tools depend on. Don't cast around it.

## Never read compiler settings from a cache key

The compilation cache key (`CacheKey`, built by `getCacheKey` / `getBuildProjectCacheKey` in
`lib/base-compiler.ts`) identifies a compilation for the cache. Its `compiler` is `{id}` and nothing else, and
anything added to it changes every cached entry. Compiler settings come from `this.compiler` in compiler code and
from `compilationInfo.compiler` in tool code, never from `key.compiler`.

## Never modify what you were given

`args`, `compilationInfo.options`, `.libraries` and `.filters` are the request's own objects, and the cache key
holds the very same ones. Pushing onto them changes the key after it was looked up, so the result is stored where
it is never found again (this is how clang-tidy compiles stopped hitting the cache). Copy first, then add:
`args = [...(args ?? []), extra]`.

## Checking a change

- `npm run ts-check` catches reads of anything but `id` through a cache key, and reads of fields not in
  `ToolCompilerInfo` through `compilationInfo.compiler`.
- In a tool's test, hand `runTool` an `args` array and assert afterwards that it is unchanged.
- `test/compilation-cache-key-tests.ts` pins what the cache key holds; a change there should be deliberate.
