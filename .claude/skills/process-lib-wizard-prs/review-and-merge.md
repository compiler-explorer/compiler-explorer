# Steps 2–5: review, verify, ask, fix and merge

Part of the `process-lib-wizard-prs` skill; paths are relative to this file's directory. The wizard is not perfect,
so **never trust its output**: this review is the main point of the skill. A *unit* is the set of pairs processed
together (one library's versions from one committer, plus any batches it depends on).

## 2. Correctness review

First read both PRs' conversations: `gh pr view <N> -R <repo> --json comments,reviews` and the inline review
comments (`gh api repos/<repo>/pulls/<N>/comments`). A maintainer may already have asked for changes, or the
contributor may have explained something (e.g. why a separate release repo is used). Don't contradict an earlier
decision or ask the contributor the same thing twice.

Then read both diffs in full (`gh pr diff <N> -R <repo>`), the library's existing entries, if any, in
`libraries.yaml` and the properties file on main, and a couple of neighbouring libraries of the same language/type
for the conventions. For a large batch, review every pair's diff; the entries differ only by version. Record each
problem as **fixable** (you will fix it in step 5) or **blocking** (it needs the contributor or me).

**Wizard bugs**: when a problem comes from the wizard itself rather than the contributor, read `wizard-bugs.md`
and follow its rule. A unit agent doesn't: it reports the defect, and the main session handles it.

**C++ libraries**: also read `windows.md` and apply its review part.

**New versions of an existing library**: if the library already exists on main and the pair only adds versions,
assume the library itself is fine: of "The library actually exists as described", check only the tag and
`check_file`, and skip the header-only question in "The type is right". Every other check below still applies,
including the link names of built libraries (targets can change between versions) and step 3.

### Collateral damage

The wizard round-trips the whole YAML file. **Key re-ordering, quote/style changes and dropped comments are
fine**, so don't flag or restore them. What matters is anything that changes meaning outside the new entry.
Compare the head semantically against the PR's merge base, with the new library or version removed from the head:
`scripts/yaml-semantic-diff.py <N> libraries/<lang>/<name>` for a new library, or
`libraries/<lang>/<name>=<target>` for a new version of an existing one. It prints `identical`, or the key paths
that differ: treat those as fixable damage.

The properties file is not round-tripped, and the wizard edits existing lines (`versions=` of an existing library).
Check every `-` line in the main diff:
- A changed `libs.<id>.versions=` line is fine if every old id is still in the new list. Order doesn't matter: don't
  flag a re-ordered list, and don't restore the old order either.
- Any other removed or changed line must be justified by the new version; anything else is damage to undo.

### The library actually exists as described

- The repo or module URL resolves, and the target tag or branch exists. Match the whole ref, not a substring
  (`grep 1.0.0` also matches `11.0.0`):
  `git ls-remote --tags --heads <url> | grep -E "refs/(tags|heads)/<target_prefix><target>(\^\{\})?$"`.
  A `v` prefix in yaml must match the upstream tag naming.
- `check_file` (if set) exists in the upstream tree at that tag.
- If yaml `repo:` and properties `url=` point at different repos (e.g. `foo/bar-release` vs `foo/bar`), find
  out why. A separate release repo can be legitimate, but confirm it.
- Plausibility: it's a real library with real content, not an empty, homework or placeholder repo, and the name
  isn't so generic that it will confuse users (`utils`, `my_library`). If in doubt, flag it to me as blocking. Don't
  reject it yourself.
- A missing LICENSE is not a finding: the author opening the PR is their consent to CE hosting it.

### Dependencies and language standard

- **C, C++, Fortran**: read the upstream build files at the tag (CMakeLists `find_package`/`*_REQUIRED_VERSION`,
  README requirements). Every library it needs must be in CE **at a compatible version** (xtensor 0.27.1 needs
  xtl ≥ 0.8.0, while CE only had xtl 0.5.3, so it failed with `xtl/xcompare.hpp: No such file`). A missing or
  too-old dependency is blocking unless an open wizard PR adds it: then that batch joins this unit (step 1).
  Otherwise it needs its own wizard PR first. Optional dependencies count too, if the library enables them by
  default or CE users are likely to (xtensor with xsimd). Users still have to select a header-only dependency
  themselves: `.dependencies=` in properties only orders the link line for built libraries.
- **Go, Rust**: dependencies in `go.mod` / `Cargo.toml` are not CE libraries and are not checked. The builders
  fetch them and ship them in the package (`go_library_builder.py` copies the module cache).
- **Minimum language standard or toolchain**, every language: C++ `-std=` (e.g. C++20), the `go` directive in
  `go.mod` (`go 1.26.0` rules out every older `gl…` compiler), `rust-version` in `Cargo.toml`. Note it in the
  report so the comment can tell users. Not blocking on its own, unless no CE compiler is new enough.

### The type is right

- **header-only**: the repo really is header-only (no `.cpp`/`.c` sources that a user must link, no CMake
  `add_library` other than INTERFACE). If it needs building, the type is wrong. That is blocking unless the fix is
  clear.
- **static/shared**: the `staticliblink`/`liblink` names in properties match the targets the build actually produces
  (look at the upstream CMakeLists). **This review is the only check of those names before merge**: the library
  builder reads them from compiler-explorer `main` on GitHub, so no local or CI build sees the PR's values.
- **trunk/branch** versions (`trunk`, `main`, `master`, any branch name): the entry must be in the language's
  `nightly:` section with `method: nightlyclone` (default branch) or `nightlybranch` (named branch), plus
  `build_type: none` for header-only, like `cppitertools`. Anything else is installed once and never updated:
  only those two methods make an installable `nightly_like` (`bin/lib/installable/git.py`), and a top-level
  entry, even with `clone_branch`, is skipped as soon as its `check_file` exists. Wizards from before
  ce-library-wizard#33 write a top-level archive entry here.
- Language specifics (go `gomod`, rust `cargo`, etc.): copy the shape of existing entries for that language.
  `lookupname` must match the conan package name the builder uses: for Go that is `go_{libname}`
  (`bin/lib/go_library_builder.py`), so a library named `go_cmp` correctly gets `go_go_cmp`.
- **Go `import_path`**: if the module root has no Go package (`golang.org/x/sys`), the entry needs `import_path:`
  pointing at a real package (`golang.org/x/sys/unix`), as `protobuf` has, or the build fails with
  `missing go.sum entry`.

### yaml and properties agree with each other and with conventions

- The library id is the same on both sides, and is only `[a-z0-9_]`. A `.` breaks it: CE splits property keys on
  dots, so `libs.mylib.git.*` is read as library `mylib` with a stray `git` key. The usual cause is a
  `.git` URL passed to the wizard, which also leaves `.git` in yaml `repo:` and properties `url=` (fixed in infra#2500
  and ce-library-wizard#35; older checkouts still do it).
- Version ids follow the file's convention: usually the version with the dots removed (C++ `0.27.1` → `0271`), but
  Go keeps the `v` (`v0.46.0` → `v0460`). Copy the existing ids of that language's file. Also check that
  every id in `versions=` has its `.version=` (and `.path=` where that type needs one).
- **Install path**: the properties `.path=` must match where ce_install installs. You check this in step 3 (it needs
  the PR's yaml). Look at the include subdir too: a `.path=` ending in `/include` only works if the repo has an
  `include/` directory, and a `v` in the path must match the yaml `target_prefix`.
- **Existing library**: if the library already exists (on main, or after an earlier merge in the same unit), the
  new version must be **added to the existing block**: append to the yaml `targets:` and to
  `libs.<id>.versions=` with its own `versions.<vid>.*` keys. Never add a second `libs.<id>.name=` block.
- Formatting: placement next to similar libraries, one blank line between blocks, and no trailing whitespace.

### CI

`gh pr checks <N> -R <repo>` on both PRs. Failing CI is blocking unless one of your fixes is meant to address it.
Pending CI is not a finding: carry on, and wait for it to go green before merging (step 5). Green CI is not
evidence of correctness either: infra CI doesn't install or build the library.

## 3. Local verification

This is the pre-merge gate. It is only needed while the wizard's output is in doubt: before running it, read
`track-record.md` to see whether it may be skipped for this pair.

All local work happens under one scratch directory, which nothing else uses: `$SCRATCH` =
`<session scratchpad>/lib-wizard`, or `lib-wizard-<unit>` for a unit agent. Test the infra PR's yaml in a worktree
there, so my checkout is not touched: `scripts/pr-worktree.sh infra <N> $SCRATCH` creates `$SCRATCH/infra-<N>`
with the python env ce_install needs.

Test what will actually merge: apply your step 5 yaml fixes in this worktree first, if there are any (for wizard
bugs, the fix agent's derived output). Never build the unfixed yaml when a known defect makes it certain to fail:
it proves nothing, and a failed Go build posts to the conan proxy (see below). For a batch, test its lowest and
highest versions, each as its own PR head plus fixes; the combined branch of step 5 doesn't exist yet and doesn't
need its own test.

**Every** ce_install command here gets `--dest $SCRATCH/opt --staging-dir $SCRATCH/staging`. Never leave `--dest` off
when `--temp-install` is used: it switches dry-run off for the install and would write to `/opt/compiler-explorer`.

- **Install path** (every type that has `.path=`; Go and Rust entries don't, skip this for them): `.path=` values
  are absolute under `/opt/compiler-explorer`, which maps to `$SCRATCH/opt`. Install with
  `bin/ce_install --dest $SCRATCH/opt --staging-dir $SCRATCH/staging install 'libraries/<lang>/<name> <target>'`,
  then `ls` the `.path` with `/opt/compiler-explorer` replaced by `$SCRATCH/opt`. The directory must exist and hold
  the expected headers/sources. This is a real install, but only into scratch.
- **Built libraries** (static/shared/cshared, go, rust, fortran…): a build that compiles for real but uploads
  nothing: `bin/ce_install --dry-run --dest $SCRATCH/opt --staging-dir $SCRATCH/staging build --temp-install
  --buildfor <compiler-id> 'libraries/<lang>/<name> <target>'`.
  **Exit code 0 is not success.** The last line must report at least `1 packages built OK`. `0 packages built OK,
  1 skipped` means nothing was tested (e.g. `is not installed, unable to build`). Treat that as a failed check and
  fix the command, don't report it as a pass. Header-only libraries log `does not have to build, skipping`, which
  is expected.

  Pick the newest release compiler that is installed locally: `scripts/newest-local-compiler.sh <lang>
  '<path-regex>'` lists the language's compilers whose executable exists, filtered to the main x86-64 toolchain by
  its install path, and prints the newest by that path. The regex matches the whole `<id> <exe>` line. C++
  `^g[0-9]+ .*/gcc-[0-9.]+/bin/g\+\+$`, C `^cg[0-9]+ .*/gcc-[0-9.]+/bin/gcc$`, Go `^gl[0-9]+ ` (cross-arch Go ids
  like `wasm_gl126` share the same binary), Rust `^r[0-9]+ `, Fortran `^gfortran[0-9]+ .*/gcc-[0-9.]+/`. Other
  languages: drop the filter (`.`), look at the output and pick similarly. Don't sort by id: ids don't sort by
  version (`gl12413` sorts after `gl1260`).

  **Go dry runs talk to the conan proxy on failure.** `go_library_builder.py` posts to `/buildfailed` without
  checking `--dry-run`. A dry run never logs in, so the post fails with `PostFailure: … buildfailed:
  UnauthorizedError: jwt malformed`. That error means *the build failed*: read the log above it for the real cause.
  Never run a local build with conan credentials set. Mention the bug in the final report until it's fixed in infra.

- **Compile check** (every C/C++ library, header-only included): compile a one-line `#include <main header>` with
  the newest local compiler (as picked above):
  `<local g++> -std=<min std> -fsyntax-only -I<lib include> [-I<dep include>…] test.cpp`.
  Header-only libraries are never built, so this is the only check that their headers compile on Linux at all
  (a `<windows.h>` include fails here, and nowhere else before users hit it). It also confirms the minimum
  standard: it must pass at the `-std=` you'll tell users, and fail one standard lower if that's the claimed
  minimum. If the library has dependencies, compile against the dependency versions CE will actually offer once
  this unit merges (install them into `$SCRATCH/opt` the same way): a clean build against a newer dependency can
  still fail for users. For other languages, the dry-run build above covers this.

Remember what this does **not** cover: the PR's link names (step 2 is the only check).

Any failure is blocking: put the relevant log lines in the report. The exception is the compile check failing for
platform reasons (a header that includes `<windows.h>`, `<conio.h>`, …) while the rest compiles: report which
headers fail and why, and let me decide. Don't mark it blocking yourself. Keep `$SCRATCH` until the final report is
done; step 5 uses the same worktrees.

## 4. Report and ask

For each unit, report:
- libraries, language, versions, author, and both PR links per pair
- every finding from step 2, marked fixable or blocking, and what you'll change for each fixable one
- the wizard bugs behind them: the issue you'll file (or the existing one) and the fix you'll publish
- the local verification result (or why it was skipped, per `track-record.md`)
- whether it needs a real library build, and with which compiler

Then ask once (AskUserQuestion; a unit agent returns the report instead): **proceed with this unit?** For a unit
with more than one pair, offer the two merge strategies as options:

- **Sequential**: merge each pair in turn, in ascending version order per library, resolving the conflicts each
  merge causes in the rest.
- **Combined branch**: merge all of the unit's PR heads into one new branch per repo on origin, open one PR
  per repo from those branches, merge them, and close the originals.

Recommend combined for 3 or more pairs, or when maintainer edits are disabled on any of them. Otherwise recommend
sequential. A yes covers everything from here to the end for that unit: filing wizard issues and publishing their
fixes, fixups, merges, the build, the comments and closing superseded PRs. It does not carry over to the next unit.

**Blocking findings** take out only the pairs that depend on them (later versions of the same library, and the
batches that need it). Ask to proceed with the rest of the unit, listing what was taken out and why. For the
pairs taken out, offer to post a review comment to the contributor describing the problems; post it only after I
say yes. If nothing is left to proceed with, only make that offer.

## 5. Fix, then merge infra and main

Do all of this in worktrees under `$SCRATCH`, never in my checkouts. For cemain, create one the same way as for
infra in step 3 (`scripts/pr-worktree.sh cemain <N> $SCRATCH`); if its pre-commit hook fails for lack of
`node_modules`, run `npm ci` there. Commit only the files you changed (`git add <file>`, then a plain
`git commit`). Never force push, never amend, never rewrite the contributor's history.

**Fixups**: if `maintainerCanModify` is true (`mcm=` in `scripts/list-wizard-prs.sh`), push a normal commit on top
of the contributor's branch, from the PR's worktree: `git push https://github.com/<owner>/<repo>.git
HEAD:<headRefName>`, with `<owner>/<repo>:<headRefName>` from the same listing. If maintainer edits are disabled,
use the combined-branch strategy (even for a single pair), or offer to ask the contributor for the fix. Fixes for
a wizard bug use the fix agent's derived output (`wizard-bugs.md`), so wait for it before pushing them.

**Merge**: once CI is green on the fixed head, merge infra, then main. Right before each merge, check
`gh pr view <N> -R <repo> --json mergeStateStatus`:
- `UNKNOWN`, often seen just after another merge, means GitHub is still computing it. Wait a few seconds and check
  again; it's not a reason to merge `main` in.
- `DIRTY` or `BEHIND` (another unit merged first; new blocks from different units usually land in the same spot
  at the end of the libs section): merge the current `origin/main` into the branch. Resolve the conflicts as in
  step 2 "Existing library", or keep both blocks if they are different libraries. Then push, wait for CI, and check
  again.
- `UNSTABLE` means a check is pending or failing: wait for CI.

Merge with `gh pr merge <N> -R compiler-explorer/infra --squash`, then the same for the main PR in
`compiler-explorer/compiler-explorer`. Never use `--admin` and never bypass branch protection. Both sides merge
before the real build because that build reads the library's properties from compiler-explorer `main`.

**Unit, sequential**: after merging one version, the remaining PRs of the unit conflict (both sides edited the same
spot) or would add duplicate blocks. For each next version, in order, in both repos:
1. In its worktree, merge (don't rebase) the current `origin/main` into it.
2. Resolve the conflicts so the result is the *existing* block with this version appended (step 2, "Existing
   library").
3. Push, wait for CI, merge.

**Unit, combined branch**: read `combined-branch.md` and follow it.

When both sides are merged, a unit agent stops here and returns what it changed; otherwise go back to `SKILL.md`
for step 6.
