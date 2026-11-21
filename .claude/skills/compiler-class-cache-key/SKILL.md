---
name: compiler-class-cache-key
description: How a compiler class in lib/compilers/ may adapt a compilation request without changing the compilation cache key. Use when writing a new compilerType, changing how an existing class handles filters, options or backend options, or reviewing such a change.
---

# Compiler classes and the cache key

`BaseCompiler.compile()` caches a compilation under the request as it arrived, plus `compiler: {id}`. ce-router
computes the same key from the request before queueing it, so it can answer repeat compiles from the cache. It knows
the compiler's id and nothing about its class. The key must therefore never depend on what a compiler class does.

`compile()` builds the key first, as a copy, and only then adapts the request to the compiler. Whatever a class does
in the hooks below changes what is compiled, never the key.

## Where to adapt a request

| To change | Override | Notes |
|---|---|---|
| Filters a request leaves out | `getDefaultFilters()` | Merged under the request's own filters, after the key. |
| Filters this compiler has to force or drop | `fixFiltersForCompiler(filters, options, files)` | E.g. always a binary (tinygo, micropython) or never one (llvm-mos). Call `super` unless you mean to replace the base rules. |
| Backend options | `fixBackendOptionsForCompiler(backendOptions)` | |
| The compiler's command line | `optionsForFilter()`, `runCompiler()` | |

## Don't

- Override `compile()` to change its arguments before `super.compile()`: that runs before the key is built, so it
  changes the key, and ce-router can't follow. Use `fixFiltersForCompiler` instead.
- Override `getCacheKey()`, or put anything but the id of the compiler into a key.
- Read compiler settings from a `CacheKey`: they come from `this.compiler`.

## Checking a change

`test/compilation-cache-key-tests.ts` pins the key for real requests, including that a class's own default filters
don't reach it. A change that moves any of those hashes changes the key, and ce-router's copy of the rules with it.
