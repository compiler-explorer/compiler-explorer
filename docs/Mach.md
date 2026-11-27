# Mach

[Mach](https://github.com/briar-systems/mach) 7 compiles a source tree from its entry module, with no manifest. For
each compilation, the adapter (`lib/compilers/mach.ts`) writes the user's entry module and any extra files from the
tree pane under `src/` in the temporary directory, then runs:

```sh
mach build src/example.mach --source example=src --source std=<stdPath> -w out --report json --debug
```

`--source example=src` binds the user's tree and `--source std=<stdPath>` the `mach-std` release the compiler is paired
with, which is read in place.

## Assembly and IR views

Mach writes each product of a build under `out/<product>/<id>/`, one file per module of the user's project. A view
reads every one of them, the entry module first and the others after it in path order. std's modules are written
beside them, under `out/<product>/std/`, and no view reads them.

- **Targets with a listing:** `--emit asm` writes `masc`'s assembly listing of each module to `out/asm/<id>/**/*.s`.
  Each module is parsed separately, since each has its own DWARF 5 `.file 0` table and `.Ltmp*` labels, and a
  secondary user file under `src/` is named as the tree pane names it.
- **Targets without one:** the compiler encodes WebAssembly and SPIR-V directly and has no text form for either, so
  the view is a disassembly of the objects `--emit obj` writes to `out/obj/<id>/**/*.o`, as it is on any target when
  the user asks for the binary object view. Objects are disassembled with `objdump`, or with `llvmObjdumper` when the
  image is not an x86 ELF. Mach 7.2.0 writes a line table into a WebAssembly object but no `.debug_info` unit to find
  it by, so `llvm-objdump` names no source line there: the view links to nothing in the editor, and the library
  functions filter, which goes by line information, hides all of it.
- **SPIR-V (`vulkan-spirv`):** built with `-k module` and disassembled with `disassemblerPath` (`spirv-dis`).
  `NonSemantic.Shader.DebugInfo.100` records map the instructions back to editor lines and columns.
- **Linked binary:** the binary filter passes `--emit exe -o out/bin/<id>` and disassembles the linked image,
  `out/bin/<id>.spv` on SPIR-V.
- **Mach IR pane:** opening it adds `--emit ir-opt` and reads the optimised, target-legalised MIRL of each module from
  `out/ir-opt/<id>/**/*.ir`. The `file` and `loc` entries of a module's `debug` table map its `at lN` instructions
  back to the editor.

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

The target list is probed, not listed. At startup the adapter reads `mach info targets` and builds a small module for
every tuple against the compiler's std: one that uses `std.runtime` and `std.print` on the systems std prints on
(`linux`, `darwin`, `windows`, `wasi`), and one that uses only `std.types` on the others (`freestanding`, `uefi`,
`vulkan`). Each tuple is built the way a compilation builds it, with `--emit asm` first and `--emit obj` if that
fails. A tuple is offered if either builds, and which one did decides whether its asm view reads the listing or the
objects. A prediscovered start reuses the answer from `cachedMachTargets`.

Tuples sharing a `<sys>-<arch>` name are told apart by appending `-<abi>`. A target marked `needs --isa` is given the
floor of its architecture and ABI unless the user passes `--isa`. A `-t` among the user's options selects the target
by any of those names and wins over the target picker.

Execution is offered on the hosted systems (`linux`, `darwin`, `windows`), where a binary for the host runs directly
and one for another host is left to Compiler Explorer's execution triple check. A request to execute on any other
target is refused without linking an executable for it.
