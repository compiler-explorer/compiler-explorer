### Using Cypress

Our frontend testing is done with cypress.

To run the tests locally:

- start a server with `npm run dev -- --language c++ --env cypress --no-local` — the `--env cypress` flag loads the fake
  compiler configuration so tests don't need a real GCC/Clang installed. The `--no-local` flag ensures your setup is
  clean of any local properties.
- in another terminal run `npx cypress open`, then choose "end to end" and then you should be able to run tests
  interactively.

### Fake compiler

Cypress tests use a `CypressCompiler` that never executes a real binary. By default it echoes the source code back as
"assembly", with each output line mapped to its source line (so line highlighting works). Tests control the output using
magic comments in the source:

```cpp
// FAKE: asm mov eax, 1     — override asm output
// FAKE: stdout hello        — set execution stdout
// FAKE: stderr error: oops  — set stderr
// FAKE: exitcode 1          — set exit/return code
// FAKE: pp int x = 42;      — canned preprocessor output
// FAKE: opt missed: ...      — optimisation remarks
// FAKE: gccdump ;; Function  — GCC dump output
```

Compiler options starting with `--fake-` also control output (useful when multiple compilers share the same source):

```
--fake-exitcode=42  --fake-stdout=hello  --fake-stderr=oops
```

Non-fake options and active filters are echoed in the output as `; Options: ...` and `; Filters: ...`, so tests can
verify that UI changes propagate to compile requests without needing real compiler behaviour.
