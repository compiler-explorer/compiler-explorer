# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands
- Build: `npm run webpack`, `npm start`
- Dev Mode: `make dev`, `make gpu-dev`
- Lint: `npm run lint` (auto-fix), `npm run lint-check` (check only)
- Type Check: `npm run ts-compile`
- Test: `npm run test` (all), `npm run test-min` (minimal)
- Test Single: `npm run test -- -t "test name"` 
- Cypress Tests: `npm run cypress`
- Pre-commit Check: `make pre-commit` or `npm run check`

## Style Guidelines
- TypeScript: Strict typing, no implicit any, no unused locals
- Formatting: 4-space indentation, 120 char line width, single quotes
- No semicolon omission, prefer const/let over var
- Client-side: ES5 JavaScript (not TypeScript) for browser compatibility
- Use Underscore.js for utility functions
- Write tests for new server-side components
- Avoid in-memory state due to clustering in production