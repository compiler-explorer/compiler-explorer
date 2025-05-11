# Running on MacOS

## Prerequisites

1. **Node.js**: Install Node.js 20 or higher
   - Using Homebrew: `brew install node`
   - Or from [nodejs.org](https://nodejs.org/)

2. **Xcode Command Line Tools**:
   - Install from the terminal: `xcode-select --install`
   - Be sure to accept the Xcode EULA (can be done by running `sudo xcodebuild -license accept` or launching Xcode once)

3. **Required Configuration Changes**:
   - Change in `etc/config/compiler-explorer.defaults.properties` the line `objdumperType=default` to `objdumperType=llvm`
   - This ensures compatibility with the macOS toolchain

## Building and Running

1. Clone the repository:
   ```bash
   git clone https://github.com/compiler-explorer/compiler-explorer.git
   cd compiler-explorer
   ```

2. Build and run:
   ```bash
   make dev
   ```

3. Access Compiler Explorer at [http://localhost:10240/](http://localhost:10240/)

## Troubleshooting

1. **Permission Issues**:
   - If you encounter permission errors with npm, try using `sudo npm install -g` or configure npm to use a directory you own

2. **Compiler Availability**:
   - By default, only Apple Clang will be available as a compiler
   - For other compilers, install with Homebrew (e.g., `brew install gcc`) and add them to your configuration

3. **Performance Issues**:
   - If you encounter slow compilation times, ensure you're not running with debug flags enabled
   - Consider adding more memory to the Node process with `NODE_OPTIONS=--max_old_space_size=4096 make dev`

## Advanced Configuration

- Create a `compiler-explorer.local.properties` file to override settings without modifying the default properties
- To add custom compilers, create a new `.local.properties` file for the language (e.g., `c++.local.properties`)

For more detailed information on configuration, see [Configuration.md](Configuration.md).
