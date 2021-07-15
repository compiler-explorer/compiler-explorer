# Adding a new compiler

This document explains how to add a new compiler to Compiler Explorer ("CE" from here on), first for a local instance, and
then how to submit PRs to get it into the main CE site.

## Configuration

Compiler configuration is done through the `etc/config/c++.*.properties` files (for C++, other languages follow the obvious pattern).
The various named configuration files are used in different contexts: for example `etc/config/c++.local.properties` take priority over
`etc/config/c++.defaults.properties`. The `local` version is ignored by git, so you can make your own personalised changes there.
The live site uses the `etc/config/c++.amazon.properties` file.

Within the file, configuration is a set of key and value pairs, separated by an `=`. Whitespace is _not_ trimmed.
Lines starting with `#` are considered comments and not parsed.
The list of compilers is set by the `compilers` key and is a list of compiler identifiers or groups, separated by colons. Group names
have an `&` prepended. As a nod to backwards compatibility with very old configurations, a path to a compiler can also be put
in the list, but that doesn't let you configure many aspects of the compiler, nor does it allow paths with colons in them (since these
are used as separators). The identifier itself is not important, but must be unique to that compiler.

An example configuration:

```
compilers=gcc620:gcc720:&clang
```

This says there are two compilers with identifiers `gcc620` and `gcc720`, and a group of compilers called `clang`. For the 
compilers, CE will look for some keys named `compiler.ID.name` and `compiler.ID.exe` (and some others, detailed later). The `ID`
is the identifier of the compiler being looked up. The `name` value is used as the human-readable compiler name shown to users,
and the `exe` should be the path name of the compiler executable.

For example:

```
compiler.gcc620.name=GCC 6.2.0
compiler.gcc620.exe=/usr/bin/gcc-6.2.0
compiler.gcc720.name=GCC 7.2.0
compiler.gcc720.exe=/usr/bin/gcc-7.2.0
```

In addition to the `name` and `exe` per-compiler configuration keys, there are also some other options. Most of them default
to sensible values for GCC-like compilers.

A group is defined similar to a list of compilers, and may contain other groups. Keys for groups start with `group.ID`.
Configuration keys applied to the group apply to all compilers in that group (unless overridden by the compiler itself). An example:

```
group.clang.compilers=clang4:clang5
group.clang.intelAsm=-mllvm -x86-asm-syntax=intel
compiler.clang4.name=Clang 4
compiler.clang4.exe=/usr/bin/clang4
compiler.clang5.name=Clang 5
compiler.clang5.exe=/usr/bin/clang5
```

### Configuration keys

Key Name | Type | Description|
---------|-------|-----|
name     | String | Human readable name of the compiler||
exe      | Path | Path to the executable|
alias    | Identifier | Another identifier for this compiler (mostly deprecated, used for backwards compatibility with very old CE URLs) |
options  | String | Additional compiler options passed to the compiler when running it |
intelAsm | String | Flags used to select intel assembly format (if not detected automatically)|
needsMulti | Boolean | Whether the compiler needs multi arch support (defaults to yes if the host has multiarch enabled)|
supportsBinary | Boolean | Whether this compiler supports compiling to binary|
supportsExecute | Boolean | Whether binary output from this compiler can be executed|
versionFlag | String | The flag to pass to the compiler to make it emit its version|
versionRe | RegExp | A regular expression used to capture the version from the version output|
compilerType | String | The name of the class handling this compiler|
interpreted | Boolean | Whether this is an interpreted language, and so the "compiler" is really an interpreter|
executionWrapper | Boolean | When executing the compiler's output, run under this script (e.g. could run under `qemu` or `mpi_run` or similar)|

The `compilerType` option is special: it refers to the Javascript class in `lib/compilers/*.js` which handles running and handling
output for this compiler type.

## Adding a new compiler locally

It should be pretty straightforward to add a compiler of your own. Create a `etc/config/c++.local.properties` file and override the
`compilers` list to include your own compiler, and its configuration.

Once you've done that, running `make` should pick up the configuration and during startup you should see your compiler being run
and its version being extracted. If you don't, check for any errors, and try running with `make EXTRA_ARGS='--debug'` to see (a lot of)
debug output.

If you're looking to add other language compilers for another language, obviously create the `etc/config/LANG.local.properties` in
the above steps, and run with `make EXTRA_ARGS='--language LANG'` (e.g. `etc/config/rust.local.properties` and
`make EXTRA_ARGS='--language Rust'`).

Test locally, and for many compilers that's probably all you need to do. Some compilers might need a few options tweaks (like
the intel asm setting, or the version flag). For a completely new compiler, you might need to define a whole new `compilerType`.
Doing so is beyond this document's scope at present, but take a look inside `lib/compilers/` to get some idea what might need
to be done.

## Adding a new compiler running remotely to your locally built compiler explorer

If you would like to have both gcc and MSVC running in the "same" compiler explorer, one option would be running gcc on your local
Linux machine and add a proxy to the MSVC compiler, which is running on a remote Window host. To achieve this, you could

* Setup compiler explorer on your Linux host as usual
* Follow [this guide](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/WindowsNative.md)
to set up another compiler explorer instance on your Windows host
* Add your Windows compiler explorer as a proxy to your Linux compiler explorer. You can simply modify your
`etc/config/c++.local.properties` on your Linux host

```
compilers=&gcc:&clang:myWindowsHost@10240
```

Yes it is the `@` symbol rather than the `:` before the port number. Restart the Linux compiler explorer, and you will be able to
see the MSVC compiler in the compiler list.

## Adding a new compiler to the live site

On the main CE website, compilers are installed into a `/opt/compiler-explorer/` directory by a set of scripts in the sister
GitHub repo: https://github.com/compiler-explorer/infra

In the `update_compilers` directory in that repository are a set of scripts that download and install binaries and compilers.
If you wish to test locally, and can create a `/opt/compiler-explorer` directory on your machine which is readable and writable by your
current user, then you can run the scripts directly. The binaries and the free compilers can be installed - the commercial compilers
live in the `install_nonfree_compilers.sh` and won't work.

If your compiler fits nicely into the harness then it should be straightforward to add it there. Anything more complex: contact the CE
authors for more help.

## Putting it all together

Hopefully that's enough to get an idea. The ideal case of a GCC-like compiler should be a pull request to add a couple of
lines to the `infra` repository to install the compiler, and a pull request to add a few lines to the `LANG.amazon.properties`
file in this repository.

If you feel like we could improve this document in any way, please contact us. We'd love to hear from you!
