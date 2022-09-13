# Adding a new library

This document explains how to add a new library to Compiler Explorer ("CE" from here on), first for a local instance,
and then how to submit PRs to get it into the main CE site.

Note that most libraries are Header-only. This is the easiest form of library to support. If the library needs to be
built, there are some caveats, best practices and good to knows. Consult the bottom of this page for details.

## Configuration

Library configurations are part of the compiler's properties, which is done through the `etc/config/c++.*.properties`
files (for C++, other languages follow the obvious pattern). The various named configuration files are used in different
contexts: for example `etc/config/c++.local.properties` take priority over `etc/config/c++.defaults.properties`. The
`local` version is ignored by git, so you can make your own personalised changes there. The live site uses the
`etc/config/c++.amazon.properties` file.

Within the file, configuration is a set of key and value pairs, separated by an `=`. Whitespace is _not_ trimmed. Lines
starting with `#` are considered comments and not parsed. The list of libraries is set by the `libs` key and is a list
of library identifiers, separated by colons. The identifier itself is not important, but must be unique to that library.

An example configuration:

```INI
libs=kvasir:boost:rangesv3
```

This says there are three libraries with identifiers `kvasir`, `boost` and `rangesv3`. CE will look for the key named
`libs.ID.versions`, `libs.ID.name` and the optionals `libs.ID.url` & `libs.ID.description`. The `ID` is the identifier
(The one we just set) of the library being looked up. The `name` key expects the human-readable name of the library
(Note that you can use spaces here!). The `versions` key expects another list, akin to the libs key itself. This time,
you have to define the available versions for each library. The `url` key expects an unescaped url, where users can go
to learn more about the library (This is usually the project's homepage, or in its absence, the GitHub repo). The
`description` key should be use as an extremely short description of the library. Usually used to spell the library's
full name in cases where the `name` key is an acronym.

For example:

```INI
libs.kvasir.name=Kvasir::mpl
libs.kvasir.versions=trunk
libs.kvasir.url=https://github.com/kvasir-io/Kvasir

libs.boost.name=Boost
libs.boost.versions=175:176
libs.boost.url=http://www.boost.org/

libs.rangesv3.name=range-v3
libs.rangesv3.description=Range library for C++11/14/17
libs.rangesv3.versions=trunk:0110
libs.rangesv3.url=https://github.com/ericniebler/range-v3
```

Now, for each declared version, CE will look for a `version` key, a human-readable string representing the corresponding
version, and `path`, a list consisting of the paths separated by colon `:` (or semicolon `;` on Windows) to add to the
inclusion path of the library. Optionally, you can provide a `libpath`, a list consisting of paths to add to your linker
path.

This would leave us with:

```INI
libs.boost.name=Boost
libs.boost.versions=175:176
libs.boost.url=http://www.boost.org/

libs.boost.versions.175.version=1.75
libs.boost.versions.175.path=/opt/compiler-explorer/libs/boost_1_75_0

libs.boost.versions.176.version=1.76
libs.boost.versions.176.path=/opt/compiler-explorer/libs/boost_1_76_0


libs.kvasir.name=Kvasir::mpl
libs.kvasir.versions=trunk
libs.kvasir.url=https://github.com/kvasir-io/Kvasir

libs.kvasir.versions.trunk.version=trunk
# Note how there are 2 paths defined for Kvasir in our case
# So that both will be added to the include paths (Example usage!)
libs.kvasir.versions.trunk.path=/opt/compiler-explorer/libs/kvasir/mpl/trunk/src/kvasir:/opt/compiler-explorer/libs/kvasir/mpl/trunk/src


libs.rangesv3.name=range-v3
libs.rangesv3.versions=trunk:0110
libs.rangesv3.url=https://github.com/ericniebler/range-v3

libs.rangesv3.versions.trunk.version=trunk
libs.rangesv3.versions.trunk.path=/opt/compiler-explorer/libs/rangesv3/trunk/include

libs.rangesv3.versions.0110.version=0.11.0
libs.rangesv3.versions.0110.path=/opt/compiler-explorer/libs/rangesv3/0.11.0/include
```

If you're adding a new library and plan to submit a PR for it, please make sure that its identifier appears in
alphabetical order in the `libs` property. You should also put all its related configuration in that same order when
defining it. This helps us keep the config manageable until further automation can be implemented. Thank you!

## Setting default libraries

The `defaultLibs` key specifies an array of libs/versions which will be enabled by default when the user visits the
site. The expected format is:

```INI
defaultLibs=libKeyA.version:libKeyB.version:libKeyC.version
```

Where `libKey` is the key of the library to be enabled by default, and `version` is the version key to load. Note that
the site won't complain if invalid key/version pairs are set. Repeating a lib key more than once is supported.

## Adding a new library locally

It should be pretty straightforward to add a library of your own. Create a `etc/config/c++.local.properties` file and
override the `libs` list to include your own library, and its configuration.

Once you've done that, running `make` should pick up the configuration, and you should be able to use them from the
library dropdown on the compiler view (The book icon)

If you're looking to add libraries for another language, obviously create the `etc/config/LANG.local.properties` in the
above steps, and run with `make EXTRA_ARGS='--language LANG` (e.g. `etc/config/rust.local.properties` and
`make EXTRA_ARGS='--language Rust'`).

Test locally, and for many compilers that's probably all you need to do. Some compilers might need a few options tweaks
(like the intel asm setting, or the version flag). For a completely new compiler, you might need to define a whole new
`compilerType`. Doing so is beyond this document's scope at present, but take a look inside `lib/compilers/` to get some
idea what might need to be done.

## Adding a new library to the live site

On the main CE website, libraries are installed into a `/opt/compiler-explorer/` directory by a set of scripts in the
sister GitHub repo: https://github.com/compiler-explorer/infra

In the `bin/yaml` directory in that repository are a set of yaml files that configure the download, install and building
of the libraries. If you wish to test locally, and can create a `/opt/compiler-explorer` directory on your machine which
is readable and writable by your current user, then you can run the scripts directly.

Example of configuring a library that is header only:

```yaml
sol2:
  type: github
  method: clone_branch
  repo: ThePhD/sol2
  check_file: include/sol/sol.hpp
  build_type: none
  targets:
    - v3.2.1
```

Example of configuring a library that is linked against:

```yaml
catch2:
  type: github
  repo: catchorg/Catch2
  build_type: cmake
  make_targets:
    - Catch2
    - Catch2WithMain
  target_prefix: v
  targets:
    - 3.0.0-preview2
```

If your library fits nicely into the harness then it should be straightforward to add it there. Anything more complex:
contact the CE authors for more help.

Remember to also add the library dependencies following the same steps. It's on you if those should also appear in the
UI.

## Adding compilers with limited library support

If you have libraries that you don't want to be shown with a compiler, you can limit the libraries per compiler. By
default, all libraries are visible for all compilers.

For example if you only want all versions of fmt, and version 0.3.0 of Ranges, you can do the following:

```ini
compiler.mycompiler.supportedLibraries=fmt:rangesv3.030
```

## Putting it all together

Hopefully that's enough to get an idea. The ideal case should be a pull request to add a couple of lines to the `infra`
repository to install the library, and a pull request to add a few lines to the `LANG.amazon.properties` file in this
repository.

If you feel like we could improve this document in any way, please contact us. We'd love to hear from you!

# Adding a library that needs to be compiled to .a or .so binaries

Supporting library binaries are a complicated matter.

For "C" shared libraries is relatively easy, is mostly a "solved problem", and the most common way of connecting
software and libraries together. OpenSSL has .so's to link against for x86-64 and x86. However, we currently do not
offer any other platforms, and it gets a lot harder if we tried to support that. Not to mention we currently do not have
hardware in the cloud for other platforms to actually execute your code.

For C++ libraries, static or shared, there is no standard or common way of building libraries. To be sure linking will
work, we have to rebuild the libraries for every compiler we support. We try to support at least x86-64, x86 and if
that's not possible - the default target of the compiler. For all llvm/clang based compilers, we also try to build the
libraries for libc++, just to be sure that doesn't give any runtime issues.

There are also some specific compiler flags that cause ABI incompatibility, but we're still looking for common cases; if
you have any use-cases of flags that causes linking or runtime errors, please let us know.

For us to have the possibility of crosscompiling with multiple compilers, it's recommended to be able to build with
CMake. CMake by default has support to provide different flags during compilation. Makefiles can provide ways for doing
the same, but often they have variables and flags that cannot be changed. If you're a library developer, please take
into account that we will need ways to set at least CC, CXX, CXXFLAGS. Be also aware that we will probably supply
-Wl-rpath's and/or -L to ensure that the library knows where to find their dependencies.

Because of the amount of combinations we need to produce, only the later tagged versions of most libraries have priority
in providing builds for. Daily trunk/master versions are out as well, until we figure out a way to efficiently provide
builds for this.
