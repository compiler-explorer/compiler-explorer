# Migrating Compiler Explorer to TOML

This document outlines a proposal for migrating Compiler Explorer's configuration system from the current `.properties`
format to TOML (Tom's Obvious, Minimal Language).

## Current System Limitations

Our current properties-based configuration system has several limitations:

1. **Awkward List Handling**: The system relies on splitting strings on `:` and `|` characters, which is error-prone and
   inflexible. This creates very long, difficult-to-read lines in configuration files.

2. **Poor Group Inheritance**: As noted in
   issue [#7150](https://github.com/compiler-explorer/compiler-explorer/issues/7150), group properties don't properly
   cascade across environment files, making configuration reuse difficult.

3. **Limited Data Structure Support**: The system can only represent flat key-value pairs without proper support for
   nested structures, arrays, or tables.

4. **Readability Issues**: Long property values (particularly lists) become unwieldy and hard to maintain, as noted in
   issue [#7341](https://github.com/compiler-explorer/compiler-explorer/issues/7341).

5. **No Format Standard**: The `.properties` format lacks standardized tooling for validation, formatting, and editor
   support.

## Why TOML?

TOML (Tom's Obvious, Minimal Language) provides several advantages:

1. **Human-Friendly**: TOML is designed to be easy to read and write, with clear semantics.

2. **Native Array Support**: TOML has built-in support for arrays, eliminating the need for custom string splitting.

3. **Table Support**: TOML's tables provide natural grouping of related properties.

4. **Widely Adopted**: Many projects use TOML, resulting in good editor support and validation tools.

5. **Backward Compatible Path**: TOML's structure makes it possible to maintain compatibility with the existing
   codebase.

6. **Type System**: TOML natively supports strings, integers, floats, booleans, dates, and arrays.

7. **Multi-line Strings**: TOML supports multi-line strings, making long values more readable.

## Array Usage Audit

The current configuration system uses different separators for different types of arrays. Below is an audit of the
common array patterns found in the configuration files:

### Colon-Separated (`:`) Arrays

| Property Pattern      | Description                  | Example                                                                        | Typical Length               |
|-----------------------|------------------------------|--------------------------------------------------------------------------------|------------------------------|
| `compilers`           | List of compiler IDs         | `compilers=&gcc:&clang`                                                        | Short to Medium (5-15 items) |
| `group.X.compilers`   | List of compilers in a group | `group.gcc.compilers=g10:g11:gdefault`                                         | Medium (5-15 items)          |
| `tools`               | List of tool IDs             | `tools=clangquerydefault:clangtidydefault:clangquery7:...`                     | Medium to Long (10-30 items) |
| `libs`                | List of library IDs          | `libs=boost:eigen:gsl:...`                                                     | Very Long (20-50+ items)     |
| `group.X.includeFlag` | List of include paths        | `group.libs.includeFlag=-isystem/path1:-isystem/path2`                         | Medium (3-10 items)          |
| `group.X.versions`    | Available versions           | `group.boost.versions=164:165:166:167:168:169:170:171:172:173:174:175`         | Medium to Long (5-20 items)  |
| `group.X.options`     | List of compiler options     | `group.clang10.options=-std=c++98:-std=c++11:-std=c++14:-std=c++17:-std=c++20` | Medium (3-10 items)          |

### Pipe-Separated (`|`) Arrays

| Property Pattern           | Description             | Example                                                              | Typical Length    |
|----------------------------|-------------------------|----------------------------------------------------------------------|-------------------|
| `ldPath`                   | Library search paths    | `ldPath=${exePath}/../lib\|${exePath}/../lib32\|${exePath}/../lib64` | Short (2-5 items) |
| `compiler.X.demanglerArgs` | Arguments for demangler | `demanglerArgs=-n\|-C\|--no-verbose`                                 | Short (2-5 items) |
| `compiler.X.objdumperArgs` | Arguments for objdumper | `objdumperArgs=-d\|--no-show-raw-insn\|--no-leading-addr`            | Short (2-5 items) |
| `group.X.libPath`          | Library paths           | `libPath=/path/to/lib1\|/path/to/lib2`                               | Short (2-5 items) |

### Key Patterns and Usage Insights

1. **Command-line Arguments**: Pipe-separated (`|`) is consistently used for command-line arguments and path lists that
   might contain colons. This is because colons often appear in paths (especially on Windows) and in command-line
   options.

2. **Entity Lists**: Colon-separated (`:`) is used for lists of entity IDs like compilers, compiler versions, or tools.
   These lists tend to be longer and are the ones most in need of better readability.

3. **Length Patterns**:
    - Pipe-separated lists tend to be shorter (2-5 items)
    - Colon-separated lists are often longer, with some (like library lists) becoming extremely long and difficult to
      maintain

4. **Particularly Problematic Examples**:
    - The `tools` property in language configs often becomes very long
    - Library version lists (`group.X.versions`) are frequently long and hard to read
    - The libs property in production files can have dozens of entries on a single line

### Examples of Particularly Long Arrays

From `c++.amazon.properties`:

```properties
tools=clangformat:clangquery:clangquerytrunk:clang-apply-replacements:clang-tidy:clang-tidy-13:clang-tidy-trunk:pahole:llvm-mcatrunk:readelf:strings:ldd:llvm-objdump:llvm-objdump-13:llvm-objdump-trunk:llvm-readobj:nm:llvm-cov-trunk:llvm-cov-13:include-what-you-use:include-what-you-use-trunk:llvm-dwarfdump-trunk:llvm-dwarfdump:x86to6502:sonarqube-gcc:sonarqube-clang:microsoft-analyzer:pvs-studio:objdump:readobj:nm-mp:llc:llc1_0:llc1_1:llc1_2:opt-trunk:bronto-trunk
```

From `compiler-explorer.amazon.properties`:

```properties
storageBucketSessions=compiler-explorer-sessions
sessionsExpirationInDays=30:40:60:180:365
```

From `c++.amazon.properties` (library versions):

```properties
group.boost.versions=164:165:166:167:168:169:170:171:172:173:174:175:176:177:178:179:180:181:182:183
```

## Migration Approach

### 1. Add TOML Support While Maintaining Backward Compatibility

1. Add a TOML parser dependency to the project.
2. Create a new configuration loader that can read both TOML and properties files.
3. Implement a compatibility layer that converts TOML structures to the current properties format internally.

### 2. Property Mapping Model

Properties would map from the current format to TOML as follows:

| Current Format              | TOML Representation                            |
|-----------------------------|------------------------------------------------|
| `key=value`                 | `key = "value"`                                |
| `key=true`                  | `key = true`                                   |
| `key=42`                    | `key = 42`                                     |
| `list=a:b:c`                | `list = ["a", "b", "c"]`                       |
| `args=a\|b\|c`              | `args = ["a", "b", "c"]`                       |
| `compiler.xyz.name=Foo`     | `[compiler.xyz]`<br>`name = "Foo"`             |
| `group.abc.compilers=x:y:z` | `[group.abc]`<br>`compilers = ["x", "y", "z"]` |

### 3. Group References

The current `&group` syntax can be mapped to TOML as follows:

```toml
# Current: compilers=&gcc:&clang
compilers = ["&gcc", "&clang"]

[group.gcc]
compilers = ["g7", "g8", "g9"]
groupName = "GCC"
```

### 4. Migration Process

1. **Phase 1: Dual Support**
    - Add TOML parser
    - Create property loader that reads both formats
    - Create TOML-to-properties converter
    - Add test suite to verify equivalent behavior

2. **Phase 2: Conversion of Existing Files**
    - Create a conversion script to transform .properties to .toml
    - Convert default configuration files first
    - Validate equivalence with test suite
    - Documentation update

3. **Phase 3: New Features**
    - Enhance properties system to leverage TOML's richer types
    - Improve group inheritance system
    - Add validation tools

4. **Phase 4: Complete Migration**
    - Deprecate .properties support
    - Full migration to TOML
    - Removal of legacy code

## Sample Conversions

### Example 1: Basic Compiler Configuration

**Current (.properties):**

```properties
compiler.g11.exe=/usr/bin/g++-11
compiler.g11.name=g++ 11.x
compiler.g11.options=-Wall -Wextra
compiler.g11.supportsBinary=true
```

**TOML:**

```toml
[compiler.g11]
exe = "/usr/bin/g++-11"
name = "g++ 11.x"
options = "-Wall -Wextra"
supportsBinary = true
```

### Example 2: Group Configuration

**Current (.properties):**

```properties
compilers=&gcc:&clang
group.gcc.compilers=g10:g11:gdefault
group.gcc.groupName=GCC
group.gcc.compilerType=gcc
group.clang.compilers=clang11:clang12:clangdefault
group.clang.intelAsm=-mllvm --x86-asm-syntax=intel
group.clang.compilerType=clang
```

**TOML:**

```toml
compilers = ["&gcc", "&clang"]

[group.gcc]
compilers = ["g10", "g11", "gdefault"]
groupName = "GCC"
compilerType = "gcc"

[group.clang]
compilers = ["clang11", "clang12", "clangdefault"]
intelAsm = "-mllvm --x86-asm-syntax=intel"
compilerType = "clang"
```

### Example 3: Tool Configuration with Long Lists

**Current (.properties):**

```properties
tools=clangquerydefault:clangtidydefault:clangquery7:clangquery8:clangquery9:clangquery10:clangquery11:clangquery12:strings:ldd:readelf:nm:llvmdwarfdumpdefault
tools.clangquerydefault.exe=/usr/bin/clang-query
tools.clangquerydefault.name=clang-query (default)
tools.clangquerydefault.type=independent
tools.clangquerydefault.class=clang-query-tool
tools.clangquerydefault.stdinHint=Query commands
```

**TOML:**

```toml
tools = [
    "clangquerydefault", "clangtidydefault",
    "clangquery7", "clangquery8", "clangquery9",
    "clangquery10", "clangquery11", "clangquery12",
    "strings", "ldd", "readelf", "nm", "llvmdwarfdumpdefault"
]

[tools.clangquerydefault]
exe = "/usr/bin/clang-query"
name = "clang-query (default)"
type = "independent"
class = "clang-query-tool"
stdinHint = "Query commands"
```

### Example 4: Command Arguments

**Current (.properties):**

```properties
compiler.clang.demanglerArgs=-n|-C|--no-verbose
```

**TOML:**

```toml
[compiler.clang]
demanglerArgs = ["-n", "-C", "--no-verbose"]
```

### Example 5: Library Path with Variable Substitution

**Current (.properties):**

```properties
ldPath=${exePath}/../lib|${exePath}/../lib32|${exePath}/../lib64
```

**TOML:**

```toml
ldPath = ["${exePath}/../lib", "${exePath}/../lib32", "${exePath}/../lib64"]
```

## Implementation Details

### TOML Parsing Library

For TypeScript/JavaScript, we can use one of:

- **@iarna/toml**: Full-featured TOML v1.0.0 parser with good TypeScript support
- **@ltd/j-toml**: TOML v1.0.0 parser with good performance
- **toml**: Simple TOML parser that's widely used

### Configuration System Changes

1. **Properties Loader**: Modify `properties.ts` to support both formats
2. **Interface Adapters**: Create adapter layer to normalize between formats
3. **Test Cases**: Create comprehensive tests to verify equivalence

### Property Resolution Logic

The hierarchical cascade system would remain largely unchanged, but would be enhanced to:

1. Better handle group property inheritance across files
2. Leverage TOML's native types
3. Provide clearer error messages for configuration issues

## Pros and Cons

### Pros

1. **Better Readability**: TOML's structure makes configuration more readable and maintainable
2. **Native Array Support**: Eliminates custom string parsing for lists
3. **Improved Structure**: Better organization of related settings
4. **Editor Support**: Better tooling and syntax highlighting
5. **Type Safety**: TOML's type system helps catch configuration errors
6. **Standardization**: Using a standard format improves maintainability

### Cons

1. **Migration Effort**: Requires converting all configuration files
2. **Learning Curve**: Team members must learn TOML (though it's designed to be simple)
3. **Backward Compatibility**: Need to maintain both parsers during transition
4. **Custom Logic**: Some CE-specific features (like `&group` references) still need custom handling

## Timeline and Resources

1. **Planning & Design**: 1-2 weeks
    - Finalize conversion specifications
    - Create test plan and compatibility tests

2. **Implementation**: 2-3 weeks
    - Implement parser integration
    - Create conversion utilities
    - Update documentation

3. **Testing & Validation**: 1-2 weeks
    - Test with various configuration scenarios
    - Verify backward compatibility

4. **Rollout**: Phased approach
    - Convert default configuration files
    - Allow users to opt-in to TOML
    - Eventually deprecate .properties

## Conclusion

Migrating to TOML offers significant benefits for maintainability and readability of Compiler Explorer's configuration.
The structured approach outlined above allows for a smooth transition while maintaining backward compatibility, with
clear improvements for both users and developers maintaining the configuration files.

The migration addresses the specific issues raised in #7150 and #7341, providing a more robust and flexible
configuration system that can grow with Compiler Explorer's needs.
