# Adding a new library

This document explains how to add a new library to Compiler Explorer ("CE" from here on), first for a local instance, and
then how to submit PRs to get it into the main CE site.

## Configuration

Library configurations are part of the compiler's properties, which is done through the `etc/config/c++.*.properties` files
(for C++, other languages follow the obvious pattern). The various named configuration files are used in different contexts:
 for example `etc/config/c++.local.properties` take priority over `etc/config/c++.defaults.properties`.
The `local` version is ignored by git, so you can make your own personalised changes there.
The live site uses the `etc/config/c++.amazon.properties` file.

Within the file, configuration is a set of key and value pairs, separated by an `=`. Whitespace is _not_ trimmed.
Lines starting with `#` are considered comments and not parsed.
The list of libraries is set by the `libs` key and is a list of library identifiers, separated by colons.
The identifier itself is not important, but must be unique to that library.

An example configuration:

```
libs=kvasir:boost:rangesv3
```

This says there are three libraries with identifiers `kvasir`, `boost` and `rangesv3`. CE will look for the key named
`libs.ID.versions`, `libs.ID.name` and the optionals `libs.ID.url` & `libs.ID.description`. The `ID` is the identifier (The one we just set) of the library being looked up.
The `name` key expects the human readable name of the library (Note that you can use spaces here!)
The `versions` key expects another list, akin to the libs key itself. This time, you have to define the available versions
for each library.
The `url` key expects an unescaped url, where users can go to learn more about the library (This is usually the project's homepage, or in its
absence, the GitHub repo)
The `description` key should be use as an extremely short description of the library. Usually used to spell the library's full name in cases where the `name` key is an acronym.

For example:

```
libs.kvasir.name=Kvasir::mpl
libs.kvasir.versions=trunk
libs.kvasir.url=https://github.com/kvasir-io/Kvasir
libs.boost.name=Boost
libs.boost.versions=164:165
libs.boost.url=http://www.boost.org/
libs.rangesv3.name=range-v3
libs.rangesv3.description=Range library for C++11/14/17
libs.rangesv3.versions=trunk:030
libs.rangesv3.url=https://github.com/ericniebler/range-v3
```

Now, for each declared version, CE will look for a `version` key, an human readable string representing the corresponding version,
and `path`, a list consisting of the paths to add to the inclusion path of the library.

This would leave us with: (Empty lines added for clarity. Please refrain from using them if you plan to PR us :D)

```
libs.kvasir.name=Kvasir::mpl
libs.kvasir.versions=trunk
libs.kvasir.url=https://github.com/kvasir-io/Kvasir

libs.kvasir.versions.trunk.version=trunk
# Note how there are 2 paths defined for Kvasir in our case (Example usage!)
libs.kvasir.versions.trunk.path=/opt/compiler-explorer/libs/kvasir/mpl/trunk/src/kvasir:/opt/compiler-explorer/libs/kvasir/mpl/trunk/src


libs.boost.name=Boost
libs.boost.versions=164:165
libs.boost.url=http://www.boost.org/

libs.boost.versions.164.version=1.64
libs.boost.versions.165.version=1.65

libs.boost.versions.164.path=/opt/compiler-explorer/libs/boost_1_64_0
libs.boost.versions.165.path=/opt/compiler-explorer/libs/boost_1_65_0


libs.rangesv3.name=range-v3
libs.rangesv3.versions=trunk:030
libs.rangesv3.url=https://github.com/ericniebler/range-v3

libs.rangesv3.versions.trunk.version=trunk
libs.rangesv3.versions.030.version=0.3.0

libs.rangesv3.versions.trunk.path=/opt/compiler-explorer/libs/rangesv3/trunk/include
libs.rangesv3.versions.030.path=/opt/compiler-explorer/libs/rangesv3/0.3.0/include
```

## Setting default libraries

The `defaultLibs` key specifies an array of libs/versions which will be enabled by default when the user visits the site.
The expected format is:
```
defaultLibs=libKeyA.version:libKeyB.version:libKeyC.version
```
Where `libKey` is the key of the library to be enabled by default, and `version` is the version key to load.
Note that the site won't complain if invalid key/version pairs are set. Repeating a lib key more than once is supported.

## Adding a new library locally

It should be pretty straightforward to add a library of your own. Create a `etc/config/c++.local.properties` file and override the
`libs` list to include your own library, and its configuration.

Once you've done that, running `make` should pick up the configuration and you should be able to use them from the library dropdown
on the compiler view (The book icon)

If you're looking to add libraries for another language, obviously create the `etc/config/LANG.local.properties` in
the above steps, and run with `make EXTRA_ARGS='--language LANG` (e.g. `etc/config/rust.local.properties` and
`make EXTRA_ARGS='--language Rust'`).

Test locally, and for many compilers that's probably all you need to do. Some compilers might need a few options tweaks (like
the intel asm setting, or the version flag). For a completely new compiler, you might need to define a whole new `compilerType`.
Doing so is beyond this document's scope at present, but take a look inside `lib/compilers/` to get some idea what might need
to be done.

## Adding a new library to the live site

On the main CE website, libraries are installed into a `/opt/compiler-explorer/` directory by a set of scripts in the sister
GitHub repo: https://github.com/compiler-explorer/infra

In the `update_compilers` directory in that repository are a set of scripts that download and install the libraries.
If you wish to test locally, and can create a `/opt/compiler-explorer` directory on your machine which is readable and writable by your
current user, then you can run the scripts directly.

If your library fits nicely into the harness then it should be straightforward to add it there. Anything more complex: contact the CE
authors for more help.

Remember to also add the library dependencies following the same steps. It's on you if those should also appear in the UI.

## Adding compilers with limited library support

If you have libraries that you don't want to be shown with a compiler, you can limit the libraries per compiler. By default all libraries are visible for all compilers.

For example if you only want all versions of fmt, and version 0.3.0 of Ranges, you can do the following:
```
compiler.mycompiler.supportedLibraries=fmt:rangesv3.030
```

## Putting it all together

Hopefully that's enough to get an idea. The ideal case should be a pull request to add a couple of
lines to the `infra` repository to install the library, and a pull request to add a few lines to the `LANG.amazon.properties`
file in this repository.
Once that's done, remember to update [the wiki](https://github.com/compiler-explorer/compiler-explorer/wiki/Installed-libraries) with the new library, adding the library in alphabetical order, with newer versions on top.

If you feel like we could improve this document in any way, please contact us. We'd love to hear from you!
