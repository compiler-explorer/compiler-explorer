# Mach

[Mach](https://github.com/briar-systems/mach) builds projects, not single files. For each compilation, the adapter
(`lib/compilers/mach.ts`) writes a small project into the temporary directory. The project has a generated
`mach.toml`, the user's sources under `src/`, and std realized under `dep/std/` by `mach dep pull`. The adapter then
runs `mach build` on the project root. The asm view is `objdump` run over the module's object file, and execution
runs the linked binary.

`mach dep pull` copies std into every compilation's project, which costs about 100 ms with std 9.4.1 (5.7 MB, 340
files). That is by design: mach reads nothing outside the project root, so every dependency is realized under
`dep/`. The copy can only get cheaper on the host side, for example by keeping the temporary directory on the same
filesystem as the std tree so the copy can hardlink or reflink.

## Diagnostics

The adapter builds with `--diagnostics=json`, under which mach writes one record per line to stderr (NDJSON, schema
1, described in mach's `doc/language/diagnostics-json.md`). `lib/parsers/mach-diagnostics.ts` reads those records
and places markers:

- **Headlines:** each diagnostic is shown as `<severity>[<key>]: <message>` followed by its location, and marked over
  its primary span with its severity. Its notes and help are added to the marker's message.
- **Related sites:** each is shown after the primary location and marked with its label.
- **Fixes:** each fix is offered as a quick fix on the headline's marker, with all of its edits, when every edit is in
  the file the marker is in.
- **Files:** spans name files from the project root, so a user source is `src/<path>` and its editor is `<path>`, as
  the project tree names it. std's files, under `dep/std/`, and the generated `mach.toml` belong to no editor and get
  no markers.
- **Columns:** mach counts columns in UTF-8 bytes and the editor in UTF-16 units, so the adapter remaps every marker
  and quick fix edit against the line it marks.
- **Failures:** a build, link or command-line failure is shown as its headline, with a marker only when it has a
  location in a user source. The closing summary becomes mach's `N errors / M warnings` tally.

A line that is not a record is shown as written. mach refuses `-v` and `-vv` beside `--diagnostics=json`, so a user
who passes either gets that refusal instead of the phase timings.

## Installing a compiler

The compiler does not ship with std. A project declares std as a dependency, so each compiler needs a copy of the std
release it was released and tested with. The adapter looks for that copy in this order:

1. `compiler.<id>.stdPath`, if set
2. `std` in the directory that holds the executable, i.e. `<dir of exe>/std`

The second is the layout both godbolt.org and the local defaults use:

```
/opt/compiler-explorer/mach-6.7.0/
├── mach          from the release tarball, mach-6.7.0-x86_64-linux.tar.gz
├── LICENSE
└── std/          briar-systems/mach-std at the release paired with this compiler (v9.4.1)
```

For a local install, unpack a release into `/opt/mach` and check out its std into `/opt/mach/std`. That matches
`etc/config/mach.defaults.properties`. If mach lives somewhere else, for example `/usr/local/bin/mach`, point
`compiler.mach.exe` at it and set `compiler.mach.stdPath` to a std checkout.

Pair each compiler with the std release it builds against. The two version numbers are independent, and std states
the compilers it accepts in its own `[project].mach`. godbolt.org offers mach 6.7.0 with std 9.4.1, the std mach
6.7.0 itself builds against. The pair was checked by building the three examples and running Hello World.

A root manifest must state the compilers it accepts in `[project].mach`. The adapter writes `mach = "^6.5"` into
every manifest it generates. That range is what the adapter needs, not the version of the compiler at hand: 6.5.0 is
the first release that writes the diagnostic records the adapter reads, so an older compiler is refused with a
diagnostic that names the range.

If a compiler has no std where the adapter looks, it logs an error that names the `stdPath` key and the compiler is
not offered.

## Targets

The target list comes from probing, not from a hard-coded list. At every startup, prediscovered or not, each compiler
takes every tuple from `mach info targets` and builds a small module that uses `std.print` for it, under the same
profile compilations use. Only the tuples that build are offered, and compilations reuse that list rather than probing
again. A compiler whose probe fails is not offered. When several tuples share a platform name, the adapter tells them apart by
appending the abi, and then the object format if that is still not enough, e.g. `linux-riscv64-lp64d`.

Two things currently exclude a tuple:

- **No freestanding targets.** std has no OS layer for `os=freestanding`, so any module that uses std beyond
  `std.types` is refused on those targets, with one diagnostic naming the capability it needs. The target picker is
  one list for every source, and the Hello World example imports `std.print`, so offering these targets would mostly
  produce refusals from std. Code that imports nothing, or only `std.types`, does build for them. They will be offered
  automatically once std builds freestanding.
- **Flat images.** `object=raw` has no debug model, and the compilation profile asks for debug information, which
  the asm view needs to map lines back to source.

The probe offers `linux-x86_64`, `linux-aarch64`, `linux-riscv64-{lp64,lp64f,lp64d}`, `darwin-x86_64`,
`darwin-aarch64` and `windows-x86_64`.

For targets other than x86, Compiler Explorer disassembles with `llvmObjdumper`, because the GNU objdump it uses for
x86 cannot read other architectures. mach's line tables are correct for those targets, but the asm view maps no lines
back to the source until the asm parser reads llvm-objdump's `; `-prefixed line records
([compiler-explorer#9128](https://github.com/compiler-explorer/compiler-explorer/pull/9128)).
