# Compiler Arguments Debugging Tool

## Overview
`compiler-args-app.ts` is a standalone debugging utility for testing and inspecting compiler argument parsing in Compiler Explorer. It allows you to run the argument parser for different compilers and see what arguments CE can extract and understand.

## Running the Tool

### Basic Usage
```bash
node --import tsx compiler-args-app.ts \
  --parser <compiler-type> \
  --exe <path-to-compiler> \
  [--padding <number>] \
  [--debug]
```

### Parameters
- `--parser <type>` (required): The compiler parser type to use
- `--exe <path>` (required): Path to the compiler executable
- `--padding <number>` (optional): Padding for output formatting (default: 40)
- `--debug` (optional): Enable debug output for troubleshooting

**Note:** Arguments should NOT use equals signs (=). Use spaces instead: `--parser gcc` not `--parser=gcc`

## Example Commands

### Debug GCC Argument Parsing
```bash
node --import tsx compiler-args-app.ts \
  --parser gcc \
  --exe /opt/compiler-explorer/gcc-14.2.0/bin/g++ \
  --debug
```

### Debug Clang with Custom Padding
```bash
node --import tsx compiler-args-app.ts \
  --parser clang \
  --exe /opt/compiler-explorer/clang-19.1.0/bin/clang++ \
  --padding 50
```

### Debug Rust Compiler
```bash
node --import tsx compiler-args-app.ts \
  --parser rust \
  --exe /opt/compiler-explorer/rust-1.80.0/bin/rustc
```

### Debug Go Compiler
```bash
node --import tsx compiler-args-app.ts \
  --parser golang \
  --exe /opt/compiler-explorer/golang-1.24.2/go/bin/go
```

### Supported Parser Types
- `gcc` - GNU Compiler Collection
- `clang` - Clang/LLVM
- `ldc` - LDC (D Language)
- `erlang` - Erlang
- `pascal` - Pascal compilers
- `ispc` - Intel SPMD Program Compiler
- `java` - Java
- `kotlin` - Kotlin
- `scala` - Scala
- `vc` - Visual C++
- `rust` - Rust
- `mrustc` - mrustc
- `num` - Nim
- `crystal` - Crystal
- `ts` - TypeScript Native
- `turboc` - Turbo C
- `toit` - Toit
- `circle` - Circle
- `ghc` - Glasgow Haskell Compiler
- `tendra` - TenDRA
- `golang` - Go
- `zig` - Zig

New parser types have to be added manually to the `compilerParsers` type list in `compiler-args-app.ts`

## Output Interpretation

The tool provides several types of information:

### 1. Available Arguments
Lists all compiler arguments that were successfully parsed, showing:
- The argument flag (e.g., `-O2`, `--std=c++20`)
- A description of what the argument does

### 2. Standard Versions (Stdvers)
Shows available language standard versions the compiler supports (e.g., C++11, C++14, C++17)

### 3. Targets
Lists available compilation targets the compiler can generate code for

### 4. Editions
Shows available editions (primarily for Rust compilers)

### 5. Compiler Capabilities
Reports on specific compiler features:
- `supportsOptOutput`: Whether optimization output is supported
- `supportsStackUsageOutput`: Whether stack usage reporting is supported
- `optPipeline`: Optimization pipeline information
- `supportsGccDump`: Whether GCC dump output is supported

### 6. Target Support Detection
The tool also reports which target specification format the compiler uses:
- `supportsTargetIs`: Uses `--target=<target>`
- `supportsTarget`: Uses `--target <target>`
- `supportsHyphenTarget`: Uses `-target <target>`
- `supportsMarch`: Uses `--march=<arch>`

## Debugging Tips

### 1. Use Debug Mode
Add `--debug` to see detailed parsing information and any errors that occur during argument extraction.

### 2. Check Parser Output
If arguments aren't being detected correctly:
- Verify the compiler executable path is correct
- Ensure the parser type matches the compiler type
- Check if the compiler requires special environment variables

### 3. Common Issues

**Empty or Missing Arguments**
- The compiler may not support the help flag format the parser expects
- Try running the compiler manually with `--help` to see its output format

**Parser Crashes**
- Enable debug mode to see the exact error
- Check that the compiler executable has execute permissions
- Ensure required libraries are available (use `ldd` to check dependencies)

**Incorrect Parser Type**
- Using wrong parser (e.g., `gcc` parser for a `clang` compiler) may work partially but miss specific features
- Always match the parser to the actual compiler type

### 4. Testing Custom Compilers
When adding support for a new compiler:
1. First run with an existing similar parser to see what works
2. Examine the raw help output to understand the format
3. Create a custom parser if needed in `lib/compilers/argument-parsers.ts`
4. Add the custom parser to the `compilerParsers` type list in `compiler-args-app.ts`

### 5. Environment Considerations
- The tool uses the current environment variables and working directory
- Some compilers may require specific environment setup (PATH, LD_LIBRARY_PATH, etc.)
- Julia compiler requires the wrapper script path to be set correctly

## Integration with CE Development

This tool is useful when:
- Adding support for new compiler versions
- Debugging why certain compiler options aren't appearing in the UI
- Understanding what arguments CE can extract from a compiler
- Testing custom argument parsers
- Verifying compiler configuration

The parsed arguments are used by CE to:
- Determine available optimization levels and flags
- Configure language standard options
- Detect supported architectures and target platforms
- Enable special compiler features based on flag availability:
  - Optimization pipeline viewer (`optPipeline`) when optimization output flags are detected
  - GCC tree/RTL dumps (`supportsGccDump`) when dump flags are found
  - Stack usage analysis when stack output flags are present
  - Intel syntax support when relevant flags are detected
  - CFG (Control Flow Graph) support based on available dump options
- Set compiler properties that control UI features and compilation behavior
