---
name: add-output-pane
description: Add a new compiler output view pane (like GHC Core, Rust MIR or Yul) end to end, from the backend produce flag to the Cypress test. Use when asked to add a new view, pane or output window to the compiler pane.
---

# Add a compiler output pane

A view pane spans about a dozen files on both sides, and a missing step usually fails silently: the button never
shows, or the pane opens empty. Copy an existing simple view rather than designing from scratch. The haskellCore view
is the reference used below: `git grep -il haskellCore` lists every file it touches, and that list is the checklist.
Below, `X`/`x` is the new view's name.

## Backend and shared types

1. `types/compiler.interfaces.ts`: add `supportsXView?: boolean` to `CompilerInfo`.
2. `types/compilation/compilation.interfaces.ts`: add `produceX?: boolean` to the request's compiler options and
   `xOutput?: ...` to `CompilationResult`.
3. Produce the output when `backendOptions.produceX && this.compiler.supportsXView`: generically in
   `lib/base-compiler.ts` (follow `makeHaskellCore`), or in the compiler class that owns it. Set `supportsXView` in
   that class's constructor or from a property.

## Frontend

4. `static/components.interfaces.ts`: add the `X_VIEW_COMPONENT_NAME` constant, the empty and populated state types,
   and the entry in the component state map. The name is stored in shortlinks, so it can never be renamed later.
5. `static/components.ts`: add `getXView()`/`getXViewWith(...)`, plus a case in `validateComponentState`. Without
   that case, loading the pane from a URL or shortlink throws, and the unknown name is reported to Sentry.
6. `static/event-map.ts`: add `xViewOpened`/`xViewClosed`.
7. `static/panes/x-view.ts` (+ `x-view.interfaces.ts`): copy `haskellcore-view.ts`. It extends `MonacoPane`, emits
   `xViewOpened` on construction and `xViewClosed` in `close()`, filters `compileResult` by compiler id, and reads
   `result.xOutput`.
8. `static/hub.ts`: import the pane, `registerComponent` it, and add the factory method.
9. `static/services/compilers.service.ts`: add `supportsXView` to the list of `CompilerInfo` fields that are kept.
10. `static/panes/compiler.ts`: add the button field, the `xViewOpen` flag, the drag source and click handler,
    `produceX: this.xViewOpen` in the request, the opened/closed handlers and their `eventHub.on`, and the
    button's disabled state and visibility (`toggle(!!this.compiler.supportsXView)`).
11. `views/templates/panes/compiler.pug`: add the `+newPaneButton("view-x", ...)` entry.
    `views/templates/templates.pug`: add `+monacopane("x")`.

## Optional

- `lib/clientstate-normalizer.ts`: only needed if the pane should be expressible through the `/clientstate/` API.
  Most views (haskellCore included) skip this; stack-usage is an example that has it.

## Tests

- `cypress/e2e/frontend.cy.ts`: add the pane to `PANE_DATA_MAP`, call `addPaneOpenTest`, and add its minimal
  `paneStates` entry. See `docs/internal/FrontendTesting.md`.
- Backend: test that the compiler class produces `xOutput` when `produceX` is set, with `exec` mocked (see the
  add-compiler-class skill).
- Open the pane in `make dev` against a compiler that supports it before calling it done.
