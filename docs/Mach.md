# Mach

[Mach](https://github.com/briar-systems/mach) builds projects, not single files. For each compilation, the adapter
(`lib/compilers/mach.ts`) writes a small project into the temporary directory. The project has a generated
`mach.toml`, the user's sources under `src/`, and std realized under `dep/std/` by `mach dep pull`. The adapter then
runs `mach build` on the project root. The asm view is `objdump` run over the module's object file, and execution
runs the linked binary.

`mach dep pull` copies std into every compilation's project, which costs about 77 ms with std 5.3.0 (5.5 MB, 303
files). That is by design: mach reads nothing outside the project root, so every dependency is realized under
`dep/`. The copy can only get cheaper on the host side, for example by keeping the temporary directory on the same
filesystem as the std tree so the copy can hardlink or reflink.

## Diagnostics

`lib/parsers/mach-diagnostics.ts` reads the diagnostics `mach build` prints and places markers:

- **Headlines:** each diagnostic's headline goes at its primary location, with the diagnostic's severity.
- **Related locations:** each gets the label under its snippet.
- **Fix edits:** each goes where its replacement lands.
- **Trailers:** `= note:` and `= help:` lines are added to the headline's message.
- **Files:** named by their path from `src/`, which is how the project tree names them. std's files belong to no
  editor and get no markers.

The parser follows the text layout `mach.cli.diagnostic` renders. Once mach can emit machine-readable diagnostics,
the adapter will read those instead.

## Installing a compiler

The compiler does not ship with std. A project declares std as a dependency, so each compiler needs a copy of the std
release it was released and tested with. The adapter looks for that copy in this order:

1. `compiler.<id>.stdPath`, if set
2. `std` in the directory that holds the executable, i.e. `<dir of exe>/std`

The second is the layout both godbolt.org and the local defaults use:

```
/opt/compiler-explorer/mach-5.4.0/
├── mach          from the release tarball, mach-5.4.0-x86_64-linux.tar.gz
├── LICENSE
└── std/          briar-systems/mach-std at the release paired with this compiler (v5.3.0)
```

For a local install, unpack a release into `/opt/mach` and check out its std into `/opt/mach/std`. That matches
`etc/config/mach.defaults.properties`. If mach lives somewhere else, for example `/usr/local/bin/mach`, point
`compiler.mach.exe` at it and set `compiler.mach.stdPath` to a std checkout.

Pair each compiler with the newest std release that builds with it. The two version numbers are independent. From std
5.2.0 on, std states the compilers it accepts in its own `[project].mach`, and older std releases state none. The
pairs godbolt.org installs were each checked by building the three examples and running Hello World:

| compiler | std |
| --- | --- |
| 5.4.0 | 5.3.0 |
| 5.3.1 | 5.3.0 |
| 5.2.1 | 5.1.0 |
| 5.1.0 | 3.2.1 |
| 5.0.4 | 3.2.1 |

godbolt.org offers the newest release and the latest patch of each older 5.x minor.

From 5.3.0, mach reads a compiler range from `[project].mach`. It warns when a project states none, and a later
release will require one. The adapter writes `mach = "^<major>.<minor>"` of the compiler into every manifest it
generates, but only for compilers from 5.3.0 on, because older ones refuse the key. A compiler with no configured
`semver` gets no range.

If a compiler has no std where the adapter looks, it logs an error that names the `stdPath` key and offers no
targets.

## Targets

The target list comes from probing, not from a hard-coded list. At startup, each compiler takes every tuple from
`mach info targets` and builds a small module that uses `std.print` for it, under the same profile compilations use.
Only the tuples that build are offered. When several tuples share a platform name, the adapter tells them apart by
appending the abi, and then the object format if that is still not enough, e.g. `linux-riscv64-lp64d`.

Two things currently exclude a tuple:

- **No freestanding targets.** std has no OS layer for `os=freestanding`, so any module that uses std beyond
  `std.types` fails inside std on those targets. The target picker is one list for every source, and the Hello World
  example imports `std.print`, so offering these targets would mostly produce errors from inside std. Code that
  imports nothing, or only `std.types`, does build for them. They will be offered automatically once std builds
  freestanding.
- **Flat images.** `object=raw` has no debug model, and the compilation profile asks for debug information, which
  the asm view needs to map lines back to source.

With mach 5.1.0 or later, the probe offers `linux-x86_64`, `linux-aarch64`, `linux-riscv64-{lp64,lp64f,lp64d}`,
`darwin-x86_64`, `darwin-aarch64` and `windows-x86_64`. mach 5.0.4 offers the same list without `windows-x86_64`,
because its COFF output has no debug model yet.

For targets other than x86, Compiler Explorer disassembles with `llvmObjdumper`, because the GNU objdump it uses for
x86 cannot read other architectures. mach's line tables are correct for those targets, but the asm view maps no lines
back to the source until the asm parser reads llvm-objdump's `; `-prefixed line records
([compiler-explorer#9128](https://github.com/compiler-explorer/compiler-explorer/pull/9128)).
