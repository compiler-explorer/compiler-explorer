# Running on MacOS

## Prerequisites

1. **Node.js**: Ensure you have Node.js 20 or higher installed
   - Installation options include Homebrew or the official installer from [nodejs.org](https://nodejs.org/)

2. **Xcode Command Line Tools**:
   - Required for compilers and build tools
   - Can be installed via `xcode-select --install`
   - Be sure to accept the Xcode EULA

## Required Configuration Changes

* Change in `etc/config/compiler-explorer.defaults.properties` the line `objdumperType=default` to `objdumperType=llvm`
* This ensures compatibility with macOS's LLVM-based toolchain

## Notes

* By default, Apple Clang will be available as a compiler
* For configuration details, see [Configuration.md](Configuration.md)
* Follow the standard setup instructions in the main [README.md](../README.md)
