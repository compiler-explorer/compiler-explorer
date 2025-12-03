# CE Properties Wizard

An interactive command-line tool for adding custom compilers to your local Compiler Explorer installation.

## Features

- **Automatic Detection**: Detects compiler type and language from the executable path
- **Auto-Discovery**: Automatically finds and adds all compilers in your PATH
- **Interactive Mode**: Guided prompts for configuration
- **Automation Support**: Command-line flags for scripting
- **Group Management**: Automatically adds compilers to appropriate groups
- **Validation**: Validates generated properties with `propscheck.py`
- **Safe Updates**: Only adds/updates, never removes existing configurations

## Requirements

The wizard requires Python 3.10+ and Poetry. The run scripts handle all setup automatically.

## Usage

### Interactive Mode

Run without arguments for a fully interactive experience:

**Linux/macOS:**
```bash
./run.sh
```

**Windows:**
```powershell
.\run.ps1
```

### Path-First Mode

Provide a compiler path to skip the first prompt:

**Linux/macOS:**
```bash
./run.sh /usr/local/bin/g++-13
```

**Windows:**
```powershell
.\run.ps1 "C:\MinGW\bin\g++.exe"
```

### Automated Mode

Use command-line flags to automate the process:

**Linux/macOS:**
```bash
./run.sh /usr/local/bin/g++-13 --yes
```

**Windows:**
```powershell
.\run.ps1 "C:\MinGW\bin\g++.exe" --yes
```

### Full Automation Example

**Linux/macOS:**
```bash
./run.sh /path/to/compiler \
  --id custom-gcc-13 \
  --name "GCC 13.2.0" \
  --group gcc \
  --options "-std=c++20" \
  --language c++ \
  --yes
```

**Windows:**
```powershell
.\run.ps1 "C:\path\to\compiler.exe" `
  --id custom-gcc-13 `
  --name "GCC 13.2.0" `
  --group gcc `
  --options "-std=c++20" `
  --language c++ `
  --yes
```

### Auto-Discovery

Automatically discover and add all compilers in your PATH:

```bash
./auto_discover_compilers.py --dry-run              # Preview what would be found
./auto_discover_compilers.py --languages c++,rust   # Add only C++ and Rust compilers
./auto_discover_compilers.py --yes                  # Add all found compilers automatically
```

### Batch Processing

Add multiple compilers with a simple loop:

**Linux/macOS:**
```bash
for compiler in /opt/compilers/*/bin/*; do
    ./run.sh "$compiler" --yes
done
```

**Windows:**
```powershell
Get-ChildItem "C:\Compilers\*\bin\*.exe" | ForEach-Object {
    .\run.ps1 $_.FullName --yes
}
```

## Command-Line Options

- `COMPILER_PATH`: Path to the compiler executable (optional in interactive mode)
- `--id`: Compiler ID (auto-generated if not specified)
- `--name`: Display name for the compiler
- `--group`: Compiler group to add to (e.g., gcc, clang)
- `--options`: Default compiler options
- `--language`: Programming language (auto-detected if not specified)
- `--yes, -y`: Skip confirmation prompts
- `--non-interactive`: Run in non-interactive mode with auto-detected values
- `--config-dir`: Path to etc/config directory (auto-detected if not specified)
- `--verify-only`: Only detect and display compiler information without making changes
- `--list-types`: List all supported compiler types and exit
- `--reorganize LANGUAGE`: Reorganize an existing properties file for the specified language
- `--validate-discovery`: Run discovery validation to verify the compiler is detected (default for local environment)
- `--env ENV`: Environment to target (local, amazon, etc.) - defaults to 'local'
- `--sdk-path`: Windows SDK base path for MSVC compilers (e.g., D:/efs/compilers/windows-kits-10)

## Supported Languages

The wizard currently supports:

**Systems Languages:**
- C++, C, CUDA
- Rust, Zig, V, Odin
- Carbon, Mojo

**Popular Compiled Languages:**
- D (DMD, LDC, GDC)
- Swift, Nim, Crystal
- Go, Kotlin, Java

**Functional Languages:**
- Haskell (GHC)
- OCaml, Scala

**.NET Languages:**
- C#, F#

**Scripting/Dynamic Languages:**
- Python, Ruby, Julia
- Dart, Elixir, Erlang

**Other Languages:**
- Fortran, Pascal, Ada
- COBOL, Assembly (NASM, GAS, YASM)

## Compiler Detection

The wizard attempts to detect compiler type by running version commands:
- GCC: `--version`
- Clang: `--version`
- Intel: `--version`
- MSVC: `/help`
- NVCC: `--version`
- Rust: `--version`
- Go: `version`
- Python: `--version`

If detection fails, you can manually specify the compiler type.

## MSVC Auto-Configuration

When adding MSVC compilers, the wizard automatically configures additional tools:

### Demangler Configuration
- **Automatic Detection**: Detects `undname.exe` from the MSVC installation path
- **Architecture Matching**: Uses the same architecture as the compiler (x64, x86, arm64)
- **Auto-Configuration**: Sets `demanglerType=win32` and `demangler=<path-to-undname.exe>`

### Objdumper Configuration
- **LLVM Detection**: Automatically detects `llvm-objdump.exe` if available in the MSVC installation
- **Conditional Setup**: Only adds objdumper configuration when `llvm-objdump.exe` is found
- **Auto-Configuration**: Sets `objdumperType=llvm` and `objdumper=<path-to-llvm-objdump.exe>`

### Windows SDK Integration
- **Interactive Prompt**: Prompts for Windows SDK path if auto-detection fails
- **Command-Line Option**: Use `--sdk-path` to specify SDK path non-interactively
- **Include/Library Paths**: Automatically configures MSVC include and library paths

### Example MSVC Usage

**Windows (Interactive):**
```powershell
.\run.ps1 "D:\efs\compilers\msvc-2022-ce\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64\cl.exe"
```

**Windows (Non-Interactive):**
```powershell
.\run.ps1 "D:\efs\compilers\msvc-2022-ce\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64\cl.exe" `
  --sdk-path "D:\efs\compilers\windows-kits-10" `
  --non-interactive
```

The wizard will automatically configure:
- Demangler: `D:/efs/compilers/msvc-2022-ce/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64/undname.exe`
- Objdumper: `D:/efs/compilers/msvc-2022-ce/VC/Tools/Llvm/x64/bin/llvm-objdump.exe` (if available)
- Windows SDK include and library paths

## Configuration Files

The wizard modifies `<language>.local.properties` files in `etc/config/`. It:
- Preserves existing content and formatting
- Creates backup files before modification
- Adds compilers to groups by default
- Ensures unique compiler IDs

## Examples

### Add a custom GCC installation

**Linux/macOS:**
```bash
./run.sh /opt/gcc-13.2.0/bin/g++
```

**Windows:**
```powershell
.\run.ps1 "C:\TDM-GCC-64\bin\g++.exe"
```

### Add a cross-compiler

**Linux/macOS:**
```bash
./run.sh /usr/bin/arm-linux-gnueabihf-g++ \
  --name "ARM GCC 11.2" \
  --group arm-gcc \
  --yes
```

**Windows:**
```powershell
.\run.ps1 "C:\arm-toolchain\bin\arm-none-eabi-g++.exe" `
  --name "ARM GCC 11.2" `
  --group arm-gcc `
  --yes
```

### Add a Python interpreter

**Linux/macOS:**
```bash
./run.sh /usr/local/bin/python3.12 --yes
```

**Windows:**
```powershell
.\run.ps1 "C:\Python312\python.exe" --yes
```

### Verify compiler detection only

**Linux/macOS:**
```bash
./run.sh /usr/bin/g++-13 --verify-only
```

**Windows:**
```powershell
.\run.ps1 "C:\MinGW\bin\g++.exe" --verify-only
```

### List all supported compiler types

**Linux/macOS:**
```bash
./run.sh --list-types
```

**Windows:**
```powershell
.\run.ps1 --list-types
```

This will output something like:
```
Detected compiler information:
  Path: /usr/bin/g++-13
  Language: C++
  Compiler Type: gcc
  Version: 13.2.0
  Semver: 13.2.0
  Suggested ID: custom-gcc-13-2-0
  Suggested Name: GCC 13.2.0
  Suggested Group: gcc
```

## Troubleshooting

### Compiler not detected
If the wizard can't detect your compiler type, it will prompt you to select one manually.

### Permission errors
Ensure you have write permissions to the `etc/config` directory.

### Validation failures
If `propscheck.py` reports errors, check the generated properties file for syntax issues.

## Development

To contribute to the wizard:

1. Format code: `./run.sh --format`
2. Check formatting: `./run.sh --format --check`
3. Run tests: `poetry run pytest` (after `poetry install`)

The `--format` flag runs black, ruff, and pytype formatters on the codebase.