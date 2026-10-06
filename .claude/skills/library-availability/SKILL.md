---
name: library-availability
description: Find out why a Compiler Explorer library version appears unavailable for a particular compiler — a library that cannot be selected, a conan package that never downloads, or an empty /app/<lib>/lib during a build. Distinguishes "no package was ever built for that compiler" from "the build was attempted and failed", and retrieves the failing build log. Use whenever a library works on some compilers but not others, or when a library looks missing or broken for a specific compiler version.
---

# Why a library is unavailable for a compiler

CE builds every library separately per compiler, architecture and stdlib. "Library X doesn't
work with compiler Y" almost always means *that particular combination* has no package — which
is a different question from whether the library is broken.

Two different services answer two different questions, and confusing them is the usual mistake:

| Source | Answers |
|---|---|
| conan server `/v1/conans/...` | what packages **exist** |
| conan **proxy** other endpoints | what was **attempted**, and why it failed |

**Absence from the conan search API is not evidence that a build was never attempted.** It can
equally mean the build ran and failed. Always check the proxy before concluding anything.

## 1. Confirm what CE actually did

In a compile API response, `downloads` lists the packages fetched. An empty `/app/<lib>/lib`
inside the build, or `downloads: []` with a library selected, means **no package for this
compiler** — not a corrupt or missing library.

Quick check: retry the same request with an older compiler. If it downloads, the library is
fine and only the newer compiler lacks a build.

## 2. List which compilers do have a package

```bash
curl -s "https://conan.compiler-explorer.com/v1/conans/<lib>/<ver>/<lib>/<ver>/search" \
 | python3 -c 'import json,sys
d=json.load(sys.stdin)
print("hashes:", len(d))
for c in sorted({(i["settings"].get("compiler"), i["settings"].get("compiler.version"),
                 i["settings"].get("arch")) for i in d.values()}, key=str): print(" ", c)'
```

The highest `compiler.version` present marks where builds stopped.

## 3. Ask the proxy whether it failed

```bash
curl -s "https://conan.compiler-explorer.com/failedbuilds/<lib>/<version>" \
 | python3 -c 'import json,sys
d=json.load(sys.stdin)
for b in d:
    if b.get("compiler")=="gcc" and b.get("arch")=="x86_64": print(b)'
```

Each record is one failed combination (`compiler`, `compiler_version`, `arch`, `libcxx`,
`commithash`). **No record for a combination that also has no package means it was never
attempted; a record means it was tried and failed.**

## 4. Read the failing build log

```bash
curl -s "https://conan.compiler-explorer.com/getlogging_forcommit/<lib>/<lib_version>/<commithash>/<compiler_version>/<arch>/libstdc%2B%2B"
```

- url-encode `libstdc++` as `libstdc%2B%2B`
- `commithash` is often just the version string — take it from the failedbuilds record
- the response is the full build script followed by the log; grep for `FAILED:`, `error:`,
  `collect2:` rather than reading it all

## Proxy endpoint reference

Base: `https://conan.compiler-explorer.com`. Source of truth:
[compiler-explorer/conanproxy](https://github.com/compiler-explorer/conanproxy) — routes are
defined in [`index.js`](https://github.com/compiler-explorer/conanproxy/blob/main/index.js).

### Availability and failures

| Endpoint | Returns |
|---|---|
| `GET /binaries/:libraryid/:version` | every built binary for the library version |
| `GET /failedbuilds/:library/:library_version` | one record per failed combination: `compiler`, `compiler_version`, `arch`, `libcxx`, `compiler_flags`, `commithash` |
| `GET /allfailedbuilds` | failed builds across all libraries |
| `GET /compilerfailurerates` | failure rate per compiler — use to spot a broadly broken compiler rather than a library-specific problem |
| `POST /hasfailedbefore` | `{response: bool}`; body `{library, library_version, compiler, compiler_version, arch, libcxx, flagcollection}` |
| `POST /whathasfailedbefore` | same, plus `commithash` — use this to get the `commithash` needed for the log endpoint |

### Build logs

| Endpoint | Returns |
|---|---|
| `GET /getlogging_forcommit/:library/:library_version/:commithash/:compiler_version/:arch/:libcxx` | full build script + log for one exact combination |
| `GET /getlogging/:library/:library_version/:arch/:dt` | logs by build timestamp |

### Human-readable views

| Endpoint | Returns |
|---|---|
| `GET /cpp_library_build_results/:library/:library_version/:commit_hash` | build results for a library across compilers (`?allcompilers=1` for the full set) |
| `GET /cpp_compiler_build_results/:compiler_id` | every library's build result for one compiler |
| `GET /annotations/:libraryid/:version` and `/annotations/:libraryid/:version/:buildhash` | build annotations |

### Library metadata

`GET /libraries`, `/libraries/cpp`, `/libraries/rust`, `/libraries/fortran`, `/libraries/go`

### Passthrough to the conan server

`/v1/*` is proxied straight through, so `GET /v1/conans/<lib>/<ver>/<lib>/<ver>/search` works on
this host too — but remember it only lists what **exists**.

Write endpoints (`/buildfailed`, `/buildsuccess`, `/clearbuildstatusforcompiler`,
`/clearbuildstatusforlibrary`, `POST /annotations/...`) are for the build infrastructure and
require auth — never call them while debugging.

## 5. Report the distinction

State which of these it is, because the remedies differ:

- **Never built for that compiler** — needs a library rebuild.
- **Build attempted and failed** — quote the actual error. Often the library's source predates
  a compiler change and needs an upstream fix or a version bump; no amount of infrastructure
  work will help.
- **Package exists but the link fails** — a different problem: check `DT_NEEDED` of the shipped
  `.so` against which libraries are actually selected, since each needed library must be
  selected separately.

## Worked example

Qt 6.7.0 looked absent for gcc 16.1. Search showed gcc builds stopping at `g144`, which alone
suggested "stale, never rebuilt". The proxy disproved that: failure records existed for
`g151`, `g152`, `g153`, `g161`, `g162` and none for `g144`. The log gave the cause —
`multiple definition of QtPrivate::IsFloatType_v<_Float16>` when linking Qt's bootstrap tools,
from a `_Float16` change in gcc 15 that Qt 6.7's headers predate. Conclusion: actively broken
on gcc >= 15.1, fixable only upstream — not an infrastructure task.
