# Mach

[Mach](https://github.com/briar-systems/mach) builds projects, not single files. For each compilation, the adapter
(`lib/compilers/mach.ts`) writes a small project into the temporary directory. The project has a generated
`mach.toml`, the user's sources under `src/`, and std realized under `dep/std/` by `mach dep pull`. The adapter then
runs `mach build` on the project root. The asm view is `objdump` run over the module's object file, and execution
runs the linked binary.

`mach dep pull` copies std into every compilation's project, which costs about 66 ms at std 3.2.0. mach does not yet
let a project use a path dependency in place. Once it does
([briar-systems/mach#3484](https://github.com/briar-systems/mach/issues/3484)), the adapter can drop the pull and the
copy.

## Installing a compiler

The compiler does not ship with std. A project declares std as a dependency, so each compiler needs a copy of the std
release it was released and tested with. The adapter looks for that copy in this order:

1. `compiler.<id>.stdPath`, if set
2. `std` in the directory that holds the executable, i.e. `<dir of exe>/std`

The second is the layout both godbolt.org and the local defaults use:

```
/opt/compiler-explorer/mach-5.2.0/
├── mach          from the release tarball, mach-5.2.0-x86_64-linux.tar.gz
├── LICENSE
└── std/          briar-systems/mach-std at the release paired with this compiler (v3.2.0)
```

For a local install, unpack a release into `/opt/mach` and check out its std into `/opt/mach/std`. That matches
`etc/config/mach.defaults.properties`. If mach lives somewhere else, for example `/usr/local/bin/mach`, point
`compiler.mach.exe` at it and set `compiler.mach.stdPath` to a std checkout.

Use the std release the compiler was released with. mach 5.2.0 pairs with std 3.2.0, and the two version numbers are
independent.

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

With mach 5.2.0 and std 3.2.0, the probe offers `linux-x86_64`, `linux-aarch64`, `linux-riscv64-{lp64,lp64f,lp64d}`,
`darwin-x86_64`, `darwin-aarch64` and `windows-x86_64`.

For targets other than x86, Compiler Explorer disassembles with `llvmObjdumper`, because the GNU objdump it uses for
x86 cannot read other architectures. mach's line tables are correct for those targets, but the asm view maps no lines
back to the source until the asm parser reads llvm-objdump's `; `-prefixed line records
([compiler-explorer#9128](https://github.com/compiler-explorer/compiler-explorer/pull/9128)).
