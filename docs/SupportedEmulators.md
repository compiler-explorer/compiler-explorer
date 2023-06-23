# Supported Emulators

## JS Client-side emulator support

These are using Javascript and/or using external websites to facilitate emulation after creating a suitable binary.

- [NES](https://github.com/compiler-explorer/jsnes-ceweb) (https://static.ce-cdn.net/jsnes-ceweb/index.html) - for images built with LLVM MOS NES or CC65 (`--target nes`)
- [JSBeeb](https://github.com/mattgodbolt/jsbeeb) (https://bbc.godbolt.org) - for binaries built with BeebAsm
- [Speccy](https://github.com/compiler-explorer/jsspeccy3) (https://static.ce-cdn.net/jsspeccy/index.html) - for `.tap` files built with Z88DK (target `+zx`)
- [Miracle](https://github.com/mattgodbolt/Miracle) (https://xania.org/miracle/miracle.html) - for `.sms` files built with Z88DK (target `+sms`)
- [Viciious](https://github.com/compiler-explorer/viciious) (https://static.ce-cdn.net/viciious/viciious.html) - for `.prg` files built with LLVM MOS C64 or CC65 (`--target c64`)

## Examples

- Color-Cycle using LLVM-MOS NES-NRom https://compiler-explorer.com/z/bhv6v9M7c
- Hello World using z88dk `+zx -lndos` https://compiler-explorer.com/z/4zh5jaov6
- Random lines using z88dk `+zx` https://compiler-explorer.com/z/h8d3dzWsr
- DStar using z88dk `+zx` https://compiler-explorer.com/z/qnE7jhnvc
- Hello World test with lines and sprites using z88dk `+sms` https://compiler-explorer.com/z/fqeGYes3b
- Star Globe demo with BeebAsm https://compiler-explorer.com/z/GjMfW3a75
- Hello World with default sprites with CC65 https://compiler-explorer.com/z/e7eGa8rKa
- Hello World with custom sprites with CC65 https://compiler-explorer.com/z/s8E3PeWfM
- Hello World with LLVM-MOS C64 https://compiler-explorer.com/z/EveEETnKT
- Bouncing balls with CC65 https://compiler-explorer.com/z/ajav6Menq
