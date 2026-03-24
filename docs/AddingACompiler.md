# Adding a new compiler

This document explains how to add a new compiler to Compiler Explorer ("CE" from here on), first for a local instance,
and then how to submit PRs to get it into the main CE site.

## Quick method: Using ce-properties-wizard

The easiest way to add a compiler to your local Compiler Explorer instance is to use the `ce-properties-wizard` tool. This interactive command-line tool automatically detects compiler information and updates your configuration files.

### Basic usage

From the Compiler Explorer root directory:

```bash
# Interactive mode - guides you through the process
etc/scripts/ce-properties-wizard/run.sh

# Path-first mode - provide compiler path directly
etc/scripts/ce-properties-wizard/run.sh /usr/bin/g++-13

# Fully automated mode - accepts all defaults
etc/scripts/ce-properties-wizard/run.sh /usr/bin/g++-13 --yes
```

### Examples

Add a custom GCC installation:
```bash
etc/scripts/ce-properties-wizard/run.sh /opt/gcc-14.2.0/bin/g++
```

Add a cross-compiler:
```bash
etc/scripts/ce-properties-wizard/run.sh /usr/bin/arm-linux-gnueabihf-g++ \
  --name "ARM GCC 11.2" \
  --group arm-gcc \
  --yes
```

The wizard will:
- Automatically detect the compiler type, version, and language
- Generate appropriate compiler IDs and display names
- Add the compiler to the correct properties file
- Suggest appropriate groups for organization
- Validate the configuration

For more options and examples, see the [ce-properties-wizard README](../etc/scripts/ce-properties-wizard/README.md).

## Manual configuration

If you need more control or want to understand how the configuration works, read on for the manual approach.

### Configuration

Compiler configuration is done through the `etc/config/c++.*.properties` files (for C++, other languages follow the
obvious pattern, replace as needed for your case).

For a comprehensive overview of the configuration system, including file hierarchy, property types, and group inheritance,
refer to [Configuration.md](Configuration.md).

Below are compiler-specific configuration details:

The list of compilers is set by the `compilers` key and is a list of compiler identifiers or groups, separated by colons. 
Group names have an `&` prepended. The identifier itself is not important, but must be unique to that compiler.

An example configuration:

```INI
compilers=gcc620:gcc720:&clang
```

This says there are two compilers with identifiers `gcc620` and `gcc720`, and a group of compilers called `clang`. For
the compilers, CE will look for some keys named `compiler.ID.name` and `compiler.ID.exe` (and some others, detailed
later). The `ID` is the identifier of the compiler being looked up. The `name` value is used as the human-readable
compiler name shown to users, and the `exe` should be the path name of the compiler executable.

For example:

```INI
compiler.gcc620.name=GCC 6.2.0
compiler.gcc620.exe=/usr/bin/gcc-6.2.0
compiler.gcc720.name=GCC 7.2.0
compiler.gcc720.exe=/usr/bin/gcc-7.2.0
```

### Don't remove or rename compilers on the public site

This applies to the public [godbolt.org](https://godbolt.org) configuration (the `*.amazon.properties` files and the
[infra](https://github.com/compiler-explorer/infra) install targets).

Shortlinks, saved sessions, and embedded widgets all refer to compilers by ID. If you remove an ID or point it at a
different version, those links break, or worse, silently show different output. So: when adding a newer patch release,
add it next to the existing one. For example, if Go 1.24.2 is `gl1242` and 1.24.13 comes out, add `gl12413` as a new
entry. Don't change what `gl1242` points to.

If a compiler really can't be kept around (say the upstream binary no longer works), use `alias` on a suitable
replacement so the old ID still resolves to something reasonable.

The same goes for the infra repo: don't remove entries from the `targets` lists in the YAML files.

Some languages use a convention where cross-architecture IDs represent the latest patch of a given major.minor series
rather than a specific patch version. For example, Go's cross-architecture entries (e.g. `386_gl124`, `arm_gl124`) are
updated in-place when a new patch release comes out (1.24.2 → 1.24.13). For the primary (amd64) entries, Go uses IDs
that include the full patch version (e.g. `gl1242`), and when a new patch comes out, the old ID is replaced with a new
one (e.g. `gl12413`). When doing this, add an `alias` on the new compiler pointing to the old ID so that existing
shared links still resolve. If you're updating a language that already follows this pattern, it's fine to continue
doing so.

In rare cases, language maintainers who are actively involved with CE have chosen to accept the breakage for
lesser-used languages. This is the exception; when in doubt, keep the old entry.

In addition to the `name` and `exe` per-compiler configuration keys, there are also some other options. Most of them
default to sensible values for GCC-like compilers.

A group is defined similar to a list of compilers, and may contain other groups. Keys for groups start with `group.ID`.
Configuration keys applied to the group apply to all compilers in that group (unless overridden by the compiler itself).
An example:

```INI
group.clang.compilers=clang4:clang5
group.clang.intelAsm=-mllvm -x86-asm-syntax=intel
compiler.clang4.name=Clang 4
compiler.clang4.exe=/usr/bin/clang4
compiler.clang5.name=Clang 5
compiler.clang5.exe=/usr/bin/clang5
```

Note about group properties: Properties defined for a group in one configuration file (e.g., `defaults`) will not be carried 
forward if that group is redefined in a higher-priority configuration file (e.g., `amazon`) without that property.

You can also use the `+=` operator to append to existing string properties. See [Configuration.md](Configuration.md#property-append-syntax) for details.

### Configuration keys

| Key Name             | Type       | Description                                                                                                                                                        |
| -------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| name                 | String     | Human readable name of the compiler                                                                                                                                |
| exe                  | Path       | Path to the executable                                                                                                                                             |
| alias                | Identifier | Another identifier for this compiler (mostly deprecated, used for backwards compatibility with very old CE URLs)                                                   |
| options              | String     | Additional compiler options passed to the compiler when running it                                                                                                 |
| intelAsm             | String     | Flags used to select intel assembly format (if not detected automatically)                                                                                         |
| needsMulti           | Boolean    | Whether the compiler needs multi arch support (defaults to yes if the host has multiarch enabled)                                                                  |
| supportsBinary       | Boolean    | Whether this compiler supports linking to binary (e.g. compile, assemble and link to final executable program)                                                     |
| supportsBinaryObject | Boolean    | Whether this compiler supports compiling to binary object (e.g. compile and assemble to binary object)                                                             |
| supportsExecute      | Boolean    | Whether binary output from this compiler can be executed                                                                                                           |
| versionFlag          | String     | The flag to pass to the compiler to make it emit its version                                                                                                       |
| versionRe            | RegExp     | A regular expression used to capture the version from the version output                                                                                           |
| compilerType         | String     | The name of the class handling this compiler                                                                                                                       |
| interpreted          | Boolean    | Whether this is an interpreted language, and so the "compiler" is really an interpreter                                                                            |
| emulated             | Boolean    | Whether the compiler's output is run via an emulator (specified by `executionWrapper`) rather than natively                                                        |
| executionWrapper     | Path       | Path to script that can execute the compiler's output (e.g. could run under `qemu` or `mpi_run` or similar)                                                        |
| executionWrapperArgs | String     | List of arguments passed to the execution wrapper (separated by `\|` character)                                                                                    |
| demangler            | String     | Path to the demangler tool                                                                                                                                         |
| demanglerArgs        | String     | List of arguments passed to the demangler binary (separated by `\|` character)                                                                                     |
| objdumper            | String     | Path to the object dump tool                                                                                                                                       |
| objdumperArgs        | String     | List of arguments passed to the object dump tool (separated by `\|` character)                                                                                     |
| instructionSet       | String     | The default set for the compiler, it will fall into that group of compilers (so you can filter by it) and get different instruction set documentation if available |

The `compilerType` option is special: it refers to the Javascript class in `lib/compilers/*.ts` which handles running
and handling output for this compiler type.

## Adding a new compiler manually

If the wizard doesn't work for your use case or you need fine-grained control, you can manually add a compiler. Create a `etc/config/c++.local.properties` file and
override the `compilers` list to include your own compiler, and its configuration.

Once you've done that, running `make` should pick up the configuration and during startup you should see your compiler
being run and its version being extracted. If you don't, check for any errors, and try running with
`make EXTRA_ARGS='--debug'` to see (a lot of) debug output.

If you're looking to add other language compilers for another language, obviously create the
`etc/config/LANG.local.properties` in the above steps, and run with `make EXTRA_ARGS='--language LANG'` (e.g.
`etc/config/rust.local.properties` and `make EXTRA_ARGS='--language Rust'`).

Test locally, and for many compilers that's probably all you need to do. Some compilers might need a few options tweaks
(like the intel asm setting, or the version flag). For a completely new compiler, you might need to define a whole new
`compilerType`. Doing so is beyond this document's scope at present, but take a look inside `lib/compilers/` to get some
idea what might need to be done.

### Generating configuration from compile_commands.json

If your project uses CMake or another build system that generates a `compile_commands.json` file, you can use the
community-maintained [compilecommands_to_compilerexplorer](https://github.com/pseyfert/compilecommands_to_compilerexplorer)
tool to automatically extract compiler paths and include directories into a `.local.properties` file.

## Adding a new compiler running remotely to your locally built compiler explorer

If you would like to have both gcc and MSVC running in the "same" compiler explorer, one option would be running gcc on
your local Linux machine and add a proxy to the MSVC compiler, which is running on a remote Window host. To achieve
this, you could

- Setup compiler explorer on your Linux host as usual
- Follow [this guide](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/WindowsNative.md) to set up
  another compiler explorer instance on your Windows host
- Add your Windows compiler explorer as a proxy to your Linux compiler explorer. You can simply modify your
  `etc/config/c++.local.properties` on your Linux host

```
compilers=&gcc:&clang:myWindowsHost@10240
```

Yes it is the `@` symbol rather than the `:` before the port number. Restart the Linux compiler explorer, and you will
be able to see the MSVC compiler in the compiler list.

## Adding a new compiler to the live site

On the main CE website, compilers are installed into `/opt/compiler-explorer/` using the `ce_install` tool from the
sister GitHub repo: https://github.com/compiler-explorer/infra

Compiler definitions are YAML-based configurations in the `bin/yaml/` directory of that repository (e.g., `cpp.yaml`,
`rust.yaml`). For many compilers, adding a new version is as simple as adding the version number to the `targets:` list
in the appropriate YAML file. See the infra repository's documentation at `docs/ce_install_yaml.md` and
`docs/installing_compilers.md` for comprehensive details on the YAML configuration format and installation process.

If you wish to test locally, create a `/opt/compiler-explorer` directory readable and writable by your user, then run
`./bin/ce_install install 'compilers/LANG/ARCH/COMPILER VERSION'` from the infra repository. Free compilers install
normally; commercial compilers marked `non-free` in the YAML won't work without proper licensing.

If your compiler fits the existing patterns it should be straightforward. Anything more complex: contact the CE authors.

## Adding a patched GCC or Clang compiler

Compiler Explorer hosts experimental branches of GCC and Clang that implement proposed C++ features. This requires
PRs to four repositories: the builder repo, compiler-workflows, infra, and this repo.

### 1. Configure the builder

Add a case block to `build/build.sh` in [clang-builder](https://github.com/compiler-explorer/clang-builder) or
[gcc-builder](https://github.com/compiler-explorer/gcc-builder).

#### Clang

[Example commit](https://github.com/compiler-explorer/clang-builder/commit/826e1e93f0dff5d83a9ac98df33b39cfbcfbf718):

```bash
p3334-trunk)
  BRANCH=p3334-cross-static
  URL=https://github.com/tal-yac/llvm-project
  VERSION=p3334-trunk-$(date +%Y%m%d)
  ;;
```

#### GCC

```bash
elif echo "${VERSION}" | grep 'lock3-contracts'; then
    VERSION=lock3-contracts-trunk-$(date +%Y%m%d)
    URL=https://github.com/lock3/gcc.git
    BRANCH=contracts
    MAJOR=13
    MAJOR_MINOR=13-trunk
    LANGUAGES=c,c++
```

### 2. Configure CI workflow

Add to [compiler-workflows](https://github.com/compiler-explorer/compiler-workflows)' `compilers.yaml` ([example](https://github.com/compiler-explorer/compiler-workflows/commit/688f0008a5f12fb976842926fd9d64d279685dd1)):

```yaml
- { image: clang, name: clang_p3334, args: p3334-trunk }
```

The `args` value must match the case label (or `if` check) in `build.sh`. Run `make build-yamls` to generate the workflow file.

### 3. Configure installation

Add to the nightly targets in [infra](https://github.com/compiler-explorer/infra)'s `bin/yaml/cpp.yaml` ([example](https://github.com/compiler-explorer/infra/commit/022371a55584fe6be4b7c24ad8e74078711c9567)):

```yaml
    nightly:
      if: nightly
      clang:
        type: nightly
        check_exe: bin/clang++ --version
        targets:
          - trunk
          - assertions-trunk
          - p3334-trunk # <-- add a line like this in the appropriate place
          - p3367-trunk
```

### 4. Configure Compiler Explorer

In `etc/config/c++.amazon.properties` ([example](https://github.com/compiler-explorer/compiler-explorer/commit/6f9cfdef90159ba20f61a807b4e113c6324b5b17)):

```ini
# Add to group compiler list
group.clangx86trunk.compilers=clang_trunk:clang_assertions_trunk:clang_p3334:...

# Configure the compiler
compiler.clang_p3334.exe=/opt/compiler-explorer/clang-p3334-trunk/bin/clang++
compiler.clang_p3334.semver=(experimental P3334)
compiler.clang_p3334.notification=Experimental cross static; see <a href="https://github.com/tal-yac/llvm-project/tree/p3334-cross-static" target="_blank" rel="noopener noreferrer">P3334<sup><small class="fas fa-external-link-alt opens-new-window" title="Opens in a new window"></small></sup></a>
```

The `notification` field creates a tooltip linking to documentation. For GCC compilers, also add `demangler`,
`objdumper`, and `isNightly=true` properties, as necessary (check some of the other compilers around for inspiration).

## Putting it all together

Hopefully that's enough to get an idea. The ideal case of a GCC-like compiler should be a pull request to add a couple
of lines to the `infra` repository to install the compiler, and a pull request to add a few lines to the
`LANG.amazon.properties` file in this repository.

If you feel like we could improve this document in any way, please contact us. We'd love to hear from you!
