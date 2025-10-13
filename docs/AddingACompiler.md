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
- Validate the configuration with `propscheck.py`

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

## Putting it all together

Hopefully that's enough to get an idea. The ideal case of a GCC-like compiler should be a pull request to add a couple
of lines to the `infra` repository to install the compiler, and a pull request to add a few lines to the
`LANG.amazon.properties` file in this repository.

If you feel like we could improve this document in any way, please contact us. We'd love to hear from you!
