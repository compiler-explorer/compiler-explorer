# Mach

[Mach](https://github.com/briar-systems/mach) 7 builds single-file and multi-file source trees directly through entry
builds. For each compilation, the adapter (`lib/compilers/mach.ts`) writes the user's entry module and any extra files
from the tree pane under `src/` in the temporary directory, then invokes:

```sh
mach build src/example.mach --source example=src --source std=<stdPath> -w out --report json --debug
```

`--source std=<stdPath>` binds the compiler's paired `mach-std` checkout in place without generating a `mach.toml` or
copying `std` into each compilation directory.

## Assembly and IR views

- **CPU targets (`x86_64`, `aarch64`, `riscv64`, `riscv32`, `riscv32e`):** Non-binary compilations pass `--emit asm` and
  read `masc`'s assembly listings under `out/asm/<id>/**/*.s`, the entry module first and any other project modules
  after it in path order. Each module is parsed separately so its DWARF 5 `.file 0` table and `.Ltmp*` labels stay
  isolated, while secondary user files under `src/` are named as the tree pane names them. Enabling the binary filter
  passes `--emit exe -o out/bin/<id>` and disassembles the linked executable with `objdump` (on x86) or `llvm-objdump`
  (on other architectures).
- **WebAssembly targets (`wasi-wasm32`, `freestanding-wasm32`):** Because WebAssembly objects bypass `masc`'s text
  builder, compilations pass `--emit obj` (or `--emit exe` when linking) and disassemble the resulting WebAssembly
  objects with `llvmObjdumper`.
- **SPIR-V (`vulkan-spirv`):** Compilations build `-k module` (`--emit obj` by default, or `--emit exe -o
  out/bin/<id>.spv` when linking a shader module), disassemble with `disassemblerPath` (`spirv-dis`), and map
  `NonSemantic.Shader.DebugInfo.100` debug records back to editor lines and columns.
- **Mach IR pane:** Opening the Mach IR pane adds `--emit ir-opt`, reading the optimised (or target-legalised) MIRL
  text from `out/ir-opt/<id>/**/*.ir` and mapping `debug` table `file` and `loc` entries on `at lN` instructions back
  to the editor.

## Diagnostics

The adapter builds with `--report json`, under which Mach writes NDJSON records (schema 3, documented in Mach's
`doc/language/report-records.md`) to stderr. `lib/parsers/mach-diagnostics.ts` reads those records and places markers:

- **Headlines:** each diagnostic is shown as `<severity>[<code>]: <message>` followed by its location, and marked over
  its primary span with its severity. Notes and help lines are folded into the marker hover message.
- **Related sites:** each is shown after the primary location and marked with its label.
- **Fixes:** each suggested fix is offered as a Monaco quick fix on the headline marker when all of its edits target
  the marked file.
- **Files and columns:** spans inside `src/` map to their editor in the project tree, while spans inside `std` get no
  editor markers. Mach counts columns in UTF-8 bytes and Monaco in UTF-16 units, so markers and quick-fix edits are
  remapped against the source lines they touch.
- **Readout records:** passing `-v` or `-vv` emits `unit` and `phase` timing records alongside diagnostics rather than
  conflicting with JSON reporting.

## Installing a compiler

Mach does not bundle `std` inside the compiler binary. Each compiler install pairs with the `mach-std` release it was
built and tested against. The adapter looks for `std` in this order:

1. `compiler.<id>.stdPath`, if set
2. `<dir of exe>/std`

Both godbolt.org and `etc/config/mach.defaults.properties` use the second layout:

```
/opt/compiler-explorer/mach-7.2.0/
├── mach          from mach-7.2.0-x86_64-linux.tar.gz
├── LICENSE
└── std/          briar-systems/mach-std at v10.1.1
```

For a local install at `/usr/local/bin/mach` or `~/.local/bin/mach`, point `compiler.<id>.exe` at the binary and set
`compiler.<id>.stdPath` to a `mach-std` checkout. If `std/mach.toml` is missing, startup logs an error naming the
`stdPath` property and drops that compiler.

## Targets and execution

At discovery time (reused via `cachedMachTargets` on prediscovered starts), the adapter reads `mach info targets` and
probes every tuple against the bundled `std`: hosted and WASI systems (`linux`, `darwin`, `windows`, `wasi`) are probed
with a `std.runtime` + `std.print` program, and bare-metal, firmware, and shader systems (`freestanding`, `uefi`,
`vulkan`) are probed with a `std.types` module. Tuples sharing a `<sys>-<arch>` name are disambiguated by appending
`-<abi>`, and targets marked `needs --isa` receive their architecture and ABI floor unless the user supplies `--isa`.

Execution is enabled only for hosted operating system targets (`linux`, `darwin`, `windows`), where native host
binaries run directly and foreign hosted binaries defer to Compiler Explorer's execution triple check. Non-hosted
targets (`freestanding`, `uefi`, `wasi`, `vulkan`) reject execution immediately without attempting to launch a
non-host image.
