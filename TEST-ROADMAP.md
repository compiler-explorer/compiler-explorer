# Compiler Explorer Test Roadmap

This document captures the current state and future plans for improving test coverage in Compiler Explorer.

## Recent Work: BaseCompiler Test Consolidation

### Consolidated Files
- base-compiler-tests.ts (original)
- base-compiler-tests-new.ts (new compilation tests)
- base-compiler-exec-tests.ts (execution tests)
- base-compiler-utils-tests.ts (utility tests)

### Test Organization
The new consolidated `base-compiler-tests.ts` organizes tests into logical sections:

1. **Basic compiler invariants**
   - Core functionality tests for version checking, include handling, etc.

2. **BaseCompiler core functionality**
   - Compilation success/failure
   - Output generation

3. **Caching behavior**
   - Cache fetching, storage, bypassing
   - Cache key generation

4. **Demangling functionality**
   - Symbol demangling
   - Demangler execution

5. **Execution functionality**
   - Code execution
   - Execution failure handling
   - Timeout handling

6. **Error handling**
   - Compilation/execution timeouts
   - Output truncation
   - Source validation
   - Memory safety errors

7. **Path handling, options, and compilation settings**
   - Path validation
   - Compiler option handling
   - File writing

8. **Execution options handling**
   - Options with spaces
   - Compiler overrides
   - Optimization output processing

9. **Environment and path handling**
   - Environment variable management
   - Path configuration

10. **Target and architecture handling**
    - Target detection
    - Architecture flags

11. **Language-specific compiler overrides**
    - Rust-specific logic
    - Other language specializations

## Recent Additions

Added tests for previously untested methods:

- **couldSupportASTDump**
  - Tests AST dump capability detection for different compiler versions

- **isCfgCompiler**
  - Tests control flow graph support detection
  - Covers different compiler types (Clang, GCC, ICC)

- **getTargetFlags/getAllPossibleTargetFlags**
  - Tests target architecture flag generation
  - Tests combinations of compiler capabilities

- **getStdverFlags/getStdVerOverrideDescription**
  - Tests standard version flag generation
  - Tests description string generation

- **sanitizeCompilerOverrides (enhanced)**
  - Added more thorough tests for environment variable handling

## Cross-Platform Considerations

- Windows-specific path handling:
  - Skip path-specific tests on Windows where necessary
  - Use platform-specific assertions
  - Path join/normalization handling

## Future Test Improvements

1. **Potential Target Areas for Additional Tests**:

   - **BaseCompiler Methods**:
     - processGccDumpOutput 
     - populatePossibleRuntimeTools
     - populatePossibleOverrides
     - initialiseLibraries
     - getCompilerResultLanguageId

   - **Compiler-Specific Classes**:
     - ClangCompiler
     - RustCompiler 
     - GccCompiler
     - LLVMCompiler

   - **Support Services**:
     - Cache implementations
     - S3 storage
     - Execution environments
     - Demanglers

2. **Testing Strategies**:

   - **Integration Testing**:
     - Test multiple compiler phases together
     - Test compiler chains (compilation → execution)

   - **Property-Based Testing**:
     - Generate random inputs to test invariants
     - Test against a wider range of inputs

   - **Mock Improvements**:
     - More realistic filesystem mocks
     - Better exec mocks for handling complex commands

3. **Improvements to Test Infrastructure**:

   - More helper utilities for compiler testing
   - Better mock management
   - More realistic test environments

## Test Improvement Project Ideas

1. **Consolidate/Improve Other Test Files**:
   - Apply same consolidation approach to other test areas
   - Reorganize by functionality rather than file structure

2. **Tools Tests**:
   - Enhance coverage for compiler tools
   - Test different tool configurations

3. **Special Language Support**:
   - Focus on language-specific compilers
   - Test language-specific features

4. **Cross-platform Experience**:
   - Improve Windows-compatibility across all tests
   - Test on different OS environments

## Connecting to this Project Later

When returning to this testing initiative later:

1. Review this document to understand the current state and next steps
2. Examine the consolidated `base-compiler-tests.ts` as a model for future test organization
3. Check CLAUDE.md for updated testing guidelines
4. Consider targets from the "Future Test Improvements" section

To continue this work with Claude:
- Show Claude this document
- Reference this as the "test refactoring project for Compiler Explorer" 
- Mention that you want to continue improving test coverage based on this roadmap

The current consolidated test file has 54 tests and provides a good foundation for future test improvements.