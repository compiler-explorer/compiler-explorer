# App.ts Migration Plan

This document outlines the plan for gradually splitting the monolithic `app.ts` file into smaller, more focused components. We will migrate functional units of code to the new `lib/app/` directory while keeping the main entry point in `app.ts`.

## Analysis of app.ts Structure

The current `app.ts` file (approximately 880 lines) can be logically divided into the following sections:

1. **Import and Setup** (lines 1-92): Includes imports, license, and base directory setup
2. **Command Line Parsing** (lines 93-237): Handles CLI options and environment setup
3. **Configuration Loading** (lines 238-320): Loads and processes configuration properties
4. **Server and Middleware Setup** (lines 321-440): Sets up web server components and static file handling
5. **URL and Routing Handlers** (lines 441-832): Creates and configures various route handlers
6. **Main and Initialization** (lines 833-880): Handles the main startup flow and signal handling

## Migration Strategy

We'll take an incremental approach to migration:

1. Start by extracting self-contained utility functions
2. Move related groups of functions into logical service modules
3. Keep integration points in the main app.ts file until later phases
4. Ensure each extraction is fully tested before proceeding to the next

## Proposed Modules

Based on the analysis, we propose the following initial modules:

### 1. `lib/app/cli.ts`
- Command-line argument parsing (lines ~93-165)
- Environment variable setup (lines ~166-195)
- Git/release info extraction (lines ~196-223)

### 2. `lib/app/config.ts`
- Configuration hierarchy setup
- Property loading and initialization
- Language filtering

### 3. `lib/app/server.ts`
- Web server setup and configuration
- Middleware configuration
- Static file handling

### 4. `lib/app/routes.ts`
- Route initialization
- Route handlers and controllers

### 5. `lib/app/render.ts`
- Template rendering functions
- Golden layout rendering
- Config processing for views

### 6. `lib/app/metrics.ts`
- Event loop monitoring
- Prometheus metrics
- Performance monitoring

### 7. `lib/app/signals.ts`
- Signal handlers
- Process management
- Shutdown procedures

## Migration Steps

### Phase 1: Initial Setup and Simple Extractions
1. ✅ Create `lib/app/` directory
2. ✅ Extract utility functions (`measureEventLoopLag`, `getFaviconFilename`, etc.)
3. Create types/interfaces for shared data

### Phase 2: Module Creation
1. ✅ Extract command-line handling
2. ✅ Extract configuration setup
3. ✅ Extract web server setup

### Phase 3: Service Extraction
1. Extract route initialization
2. Extract rendering functions
3. Extract metrics and monitoring

### Phase 4: Integration
1. Connect extracted modules
2. Refactor app.ts to use new modules
3. Ensure main entry point remains functional

### Phase 5: Final Cleanup
1. Improve test coverage for new modules
2. Add documentation
3. Clean up remaining technical debt

## Testing Strategy

For each extraction:
1. Create unit tests for the extracted code
2. Ensure end-to-end functionality remains intact
3. Run existing tests to verify no regressions

This approach ensures we can safely refactor while maintaining full functionality.