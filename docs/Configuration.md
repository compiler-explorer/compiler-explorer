# Compiler Explorer Configuration System

This document describes the configuration system used by Compiler Explorer, which is based on `.properties` files that
follow a hierarchical structure with inheritance and specialized handling for various property types.

## Configuration File Structure

Compiler Explorer uses `.properties` files located in the `etc/config/` directory for all its configuration needs. These
files follow a simple key-value format:

```
key=value
```

Comments are lines that start with a hash character, and are not processed:

```
# This is a comment
key=value # This is also a comment
```

### File Naming Convention

Configuration files follow a specific naming pattern:

```
CATEGORY.ENVIRONMENT.properties
```

Where:

- `CATEGORY` represents a configuration category (like `c++`, `rust`, `compiler-explorer`, `aws`)
- `ENVIRONMENT` represents where the configuration will be used (like `defaults`, `amazon`, `local`)

For example:

- `c++.defaults.properties` - Default C++ configuration
- `c++.amazon.properties` - C++ configuration for the Amazon environment (used in production)
- `c++.local.properties` - Local C++ configuration (ignored by git, for personal settings)

## Configuration Hierarchy

The system loads configuration files in a specific order, creating a cascade of settings. The actual hierarchy used for
property resolution, from lowest to highest priority, is:

1. `defaults` - Base configuration that applies to all environments
2. Environment-specific settings (such as `dev`, `beta`, `staging`, `amazon`)
3. Platform-specific environment settings (like `dev.linux`, `staging.darwin`)
4. Platform-specific settings (like `linux`, `darwin`, `win32`)
5. Host-specific settings (using the machine's hostname)
6. `local` - User-specific configuration that overrides all others (can be disabled with the `--no-local` flag)

For example, if you're running on a Linux machine named "myserver" in the "staging" environment, a property would be
looked up in this order:

1. `etc/config/category.defaults.properties`
2. `etc/config/category.staging.properties`
3. `etc/config/category.staging.linux.properties`
4. `etc/config/category.linux.properties`
5. `etc/config/category.myserver.properties`
6. `etc/config/category.local.properties`

This means if the same property is defined in multiple files, the one from the most specific environment will be used.

### Important Note on Group Properties

The hierarchy/cascade only works for top-level properties. Group properties (discussed below) do not inherit across
different environment files. That is, if you define a group property in `c++.defaults.properties`, but that group is
defined without that property in `c++.amazon.properties`, the property will not be carried forward to the amazon
environment.

## Property Types

While all properties are stored as strings in the files, the system automatically converts values to appropriate types
using the `toProperty` function:

1. **Strings** - The default type for all properties
   ```
   compiler.clang.name=Clang
   ```

2. **Booleans** - Values `true`, `yes`, `false`, `no` are converted to boolean values
   ```
   compiler.clang.supportsBinary=true
   ```

3. **Numbers** - Numeric strings are converted to integers or floats
   ```
   compiler.clang.timeout=10
   ```

4. **Special Version Properties** - Properties ending with `.version` or `.semver` are never converted to numbers, even
   if they look like numbers. This preserves version formatting with leading zeros or other special characters:
   ```
   compiler.gcc123.version=9.0.0    # Remains the string "9.0.0" instead of becoming a number
   ```

## Lists and Separators

Many properties in Compiler Explorer represent lists of values. These use specific separators:

1. **Colon-separated lists** (`:`) - Most commonly used for lists of identifiers, compiler names, etc.
   ```
   compilers=gcc:clang:msvc
   ```

2. **Pipe-separated lists** (`|`) - Used for argument lists, particularly when options might contain colons
   ```
   compiler.clang.demanglerArgs=-n|-C|--no-verbose
   ```

## Compiler and Group Configuration

### Basic Compiler Configuration

Compilers are configured using properties with the `compiler.ID` prefix:

```
compiler.gcc.name=GCC
compiler.gcc.exe=/usr/bin/gcc
compiler.gcc.options=-Wall
```

### Group System with &IDENTIFIER

A key feature of the configuration system is the ability to define groups of compilers with shared settings using `&`
syntax:

```
compilers=&gcc:&clang:specific_compiler

group.gcc.compilers=gcc7:gcc8:gcc9
group.gcc.groupName=GCC
group.gcc.supportsBinary=true

group.clang.compilers=clang9:clang10
group.clang.groupName=Clang
group.clang.intelAsm=-mllvm --x86-asm-syntax=intel
```

In this example:

1. The `compilers` list includes two groups (`&gcc` and `&clang`) and one individual compiler
2. Each group defines its own list of compilers
3. Properties set at the group level (like `supportsBinary` or `intelAsm`) are inherited by all compilers in that group

Groups can contain other groups by using the `&` syntax within a group's compilers list:

```
group.newer.compilers=&clang:&gcc
group.newer.groupName=Modern Compilers
```

### Property Inheritance

Properties are inherited from groups to individual compilers. If both a group and a compiler define the same property,
the compiler's value takes precedence. This allows for group-wide defaults with compiler-specific overrides.
**There are some notable exceptions to this rule**: some unique-per-compiler properties are not inherited from groups,
but we hope to fix this. See [this GitHub issue](https://github.com/compiler-explorer/compiler-explorer/issues/7150).

## Configuration Keys

Common configuration keys include:

| Key Name       | Type    | Description                                         |
|----------------|---------|-----------------------------------------------------|
| name           | String  | Human-readable name displayed to users              |
| exe            | Path    | Path to the executable                              |
| options        | String  | Default compiler options                            |
| compilerType   | String  | Refers to the handler class in `lib/compilers/*.ts` |
| intelAsm       | String  | Flags for Intel assembly syntax                     |
| supportsX      | Boolean | Capability flags for various features               |
| versionFlag    | String  | Flag to pass to compiler to get version             |
| demanglerArgs  | String  | Arguments for the demangler (pipe-separated)        |
| objdumperArgs  | String  | Arguments for the object dumper (pipe-separated)    |
| instructionSet | String  | Default instruction set for the compiler            |

## Variable Substitution

Some properties support variable substitution to make configuration more flexible. The most common variables are:

### Path Variables

1. **${exePath}**: Replaced with the directory path of the compiler executable
   ```
   ldPath=/opt/compiler-explorer/lib/|${exePath}/../lib/
   ```

2. **${ceToolsPath}**: Replaced with the path to the Compiler Explorer tools directory
   ```
   demangler=${ceToolsPath}/demangler
   ```

### Special Properties for Environment Variables

Environment variables can be configured using a special format:

```
envVars=VAR1=value1:VAR2=value2
```

This is parsed and converted into environment variable key-value pairs that are passed to the compiler when it's
executed.

### Path Lists

Multiple types of path lists exist in the system, mainly using two separator styles:

1. **Colon-separated**: Used for general lists (compilers, versions, etc.)
   ```
   compilers=gcc:clang:msvc
   ```

2. **Pipe-separated**: Used for path lists and command arguments
   ```
   ldPath=/path/to/lib1|/path/to/lib2
   demanglerArgs=-n|-C|--no-verbose
   ```

Note that path-style properties often support variable substitution as shown above.

## Advanced Configuration Features

### Property Debugging

You can enable detailed debugging of property resolution by using the `--prop-debug` flag when starting Compiler
Explorer. This shows every property lookup, including where properties are being overridden and which configuration
source they come from.

### Remote Compiler Configuration

Compiler Explorer supports defining remote compilers using a special syntax in the configuration:

```
compilers=local1:local2:remote@hostname:port
```

When a compiler ID contains the `@` symbol followed by a hostname and port, it is interpreted as a remote compiler. This
allows running compilers on separate machines and accessing them through the main Compiler Explorer instance.

### Orphaned Property Detection

The system has built-in detection for "orphaned" properties - configurations for compilers or groups that are not
referenced anywhere. This helps maintain configuration cleanliness by identifying unused configuration entries.

## Configuration System Implementation

The configuration system is implemented primarily in the following files:

- `lib/properties.ts` - Core implementation of the properties system
- `lib/properties.interfaces.ts` - TypeScript interfaces for the properties system
- `lib/compiler-finder.ts` - Handles compiler discovery and group expansion
- `lib/utils.ts` - Contains property value conversion functions

## Debugging Configuration

If you need to troubleshoot configuration issues, you can run Compiler Explorer with debug output:

```
make EXTRA_ARGS='--prop-debug' dev
```

This will show detailed logs about property resolution, including which properties are being overridden and from which
source.
