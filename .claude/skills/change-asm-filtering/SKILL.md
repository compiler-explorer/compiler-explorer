---
name: change-asm-filtering
description: Change how compiler assembly output is filtered (labels, directives, comments, library code, binary mode, source line mapping), and update the golden test files. Use for asm filter bugs or changes to lib/parsers/asm-parser*.ts.
---

# Change assembly filtering

## Two parsers, one behaviour

- `lib/parsers/asm-parser.ts` (`AsmParser.process`) and its subclasses next to it (`asm-parser-vc.ts`, `-ptx`,
  `-spirv`, ...) are the TypeScript parsers. Compiler constructors pick them by assigning `this.asm`.
- On godbolt.org, the languages with `externalparser=CEAsmParser` in their `etc/config/<lang>.amazon.properties`
  go through the C++ [asm-parser](https://github.com/compiler-explorer/asm-parser) binary instead
  (`lib/external-parsers/`). Those include C, C++ and Rust: `git grep -l '^externalparser=' etc/config` lists them.
  Dev and local installs don't set it, so they always use the TypeScript parser.

So a fix only in the TypeScript parser changes what you see in `make dev` but not what production users see for
C/C++/Rust. If the bug report comes from godbolt.org for one of those languages, the fix (or at least an issue)
belongs in the asm-parser repo too. Mention this in the PR.

`ExternalParserBase` maps filters to the binary's CLI flags. A new filter option has to be added there as well.

## Golden tests

- Inputs live in `test/filters-cases/*.asm`. The expected output for each filter combination is next to them as
  `<name>.asm.<filters>.json`.
- `processAsm` in `test/utils.ts` picks the parser from the file name (`sass-`, `ptx-`, `ca65-`, `cc65-`, `ewarm-`,
  ...) or contents (`Microsoft` → VC). Files ending `-bin.asm` also get the binary-mode variants.
- To add a regression case, drop the minimal `.asm` reproducing it into `test/filters-cases/`, then regenerate:
  `npx vitest run -u test/filter-tests.ts`
- Review the regenerated `.json` diffs one by one. A change to the parser often shifts many goldens, so check that
  each change is intended and not a side effect.
- These tests are skipped with `SKIP_EXPENSIVE_TESTS=true` (`npm run test-min`, the pre-commit hook) and on
  Windows/macOS. Run them explicitly.
- `test/demangle-cases/` follows the same `-u` flow for demangler output.
