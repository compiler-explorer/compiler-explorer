---
name: process-lib-wizard-prs
description: Review, fix, merge and build library PRs created by ce-lib-wizard (paired infra libraries.yaml + compiler-explorer properties PRs). Use when asked to process, review or merge ce-lib-wizard / "Add <lang> library" PRs.
---

# Process ce-lib-wizard PRs

[ce-library-wizard](https://github.com/compiler-explorer/ce-library-wizard) opens a **pair** of PRs per library
version:

- **infra**: `bin/yaml/libraries.yaml` in compiler-explorer/infra (how to install/build the library)
- **main**: `etc/config/<lang>.amazon.properties` in compiler-explorer/compiler-explorer (how CE exposes it). Its body
  links the infra PR as `Related PR: https://github.com/compiler-explorer/infra/pull/<N>`. The link is one-way: the
  infra body has no link back.

Both bodies end with `_PR created with [ce-lib-wizard](...)_`. Only process PRs with that footer. The wizard makes
exactly one commit per PR (`Add <lang> library <name> <version>`). Any further commit, other than a merge of `main`
into the branch (GitHub's "Update branch"), is someone's fixup.

The wizard is not perfect, so **never trust its output**: the correctness review in step 2 is the main point of this
skill. Its job is to catch what the wizard got wrong and fix it before anything merges.

**Terms.** A *batch* is all pairs from one committer for one library (one pair per version). A *unit* is one batch,
or several batches that depend on each other and so must be processed together (xtl and xtensor, where xtensor
needs the new xtl). Choices and approvals are per unit.

Flow per unit: review → local test build → report and ask → fix → merge infra → merge main → real build →
comment. The stops where you ask me are:
- "Decide what to process", once per run;
- the approval per unit after the review (step 4);
- offers I must accept first: a review comment to a contributor, asking a contributor for a fix, closing a stale PR.

After a unit's approval, run it to the end without asking again, but a pair that fails a gate is skipped and
reported, never forced through.

## Decide what to process

Run step 1 first to find and group the pairs. If there is more than one unit, stop before step 2 and show me the
units (committer, libraries, versions, PR numbers, and why batches were joined into a unit). Then ask
(AskUserQuestion) which to do:

- **One unit first**: pick it, process it to the end, then come back to this question for the rest.
- **All at once, in separate agents**: one background agent per unit.

Everything doesn't have to be done in one go. Units I don't pick are left open for a later run; just list them in
the final report.

**Separate agents** (Agent tool, one per unit, run in the background):
- Before spawning, fetch every chosen PR head once yourself (step 3's `fetch` line), so agents don't all fetch in
  my checkouts at the same moment. If an agent's later fetch fails with `cannot lock ref`, it retries.
- Brief each agent with this skill's path, its unit's PR numbers, a unit name (e.g. `jdoe-zap`), and these
  rules. It runs steps 2–3 and returns the step 4 report **instead of asking**: subagents can't ask me. You ask me
  per unit, as step 4 describes, and send the answer back to that agent (SendMessage) so it carries on with step 5.
- The agent stops after its merges and returns what it changed. You run steps 6–8 for that unit yourself, one unit
  at a time, so only one build dispatch is in flight while you look for its run id.
- Each agent works only in its own scratch directory, `$SCRATCH` = `<session scratchpad>/lib-wizard-<unit>`, with
  its own worktrees of **both** repos there (step 3). My checkouts are never switched or committed to.
- Wizard bugs are handled by you, not the unit agents: they report defects, you apply "Rule for wizard bugs" once
  per bug (one fix agent, one issue), and forward the derived output to every unit it affects.

## Wizard bugs

The step 2 checks catch wizard bugs as well as contributor mistakes. Be aware that a bug may have been fixed since
the PR was made: contributors run their own checkout, which can be older than the fix.

**Rule for wizard bugs.** Each wizard defect you find is handled like this. First search
the wizard's issues, closed ones included (`gh issue list -R compiler-explorer/ce-library-wizard --state all
--search <keywords>`). A closed issue means the PR came from a wizard older than the fix: fix the PR by hand and
link the issue in the report, with no fix agent and no new issue. Otherwise:

1. **Before approval, locally only.** Start a fix agent (Agent tool, in the background) so the review carries on
   meanwhile. It works in its own scratch directory, in a worktree of `/opt/compiler-explorer/ce-library-wizard` off
   a freshly fetched `origin/main`. It fixes the bug, tests it, and derives the correct output: it runs the fixed
   wizard with `--dry-run --build-test=no -y` and the non-interactive flags (`--lang`, `--lib`, `--ver`, …) for every
   affected library and version, and returns the yaml entry and properties block it now generates. The agent
   commits and pushes nothing yet.
2. **Use the derived output** in step 3 (test it) and step 5 (fix the PRs by hand). Compare it with your own step 2
   fix: if they differ, find out which is right before testing or pushing anything. Affected pairs wait for it; the
   rest of the unit doesn't.
3. **After approval**, which covers this: file the bug on compiler-explorer/ce-library-wizard unless an open issue
   already covers it, with the cause in the wizard's code and links to the affected PRs. Then tell the fix agent
   (SendMessage) to push its branch to origin and open a PR that closes the issue. It doesn't merge it.

List the planned issue and fix in the step 4 report. Don't edit this skill: anything you think it should say about
wizard bugs goes in the final report for me to decide.

## 1. Find the pairs

Run `scripts/list-wizard-prs.sh` (paths in this skill are relative to its directory). It lists the open wizard PRs
in both repos, one line each, with the fields the later steps need and the main PR's `Related PR:` link.

GitHub search indexing lags by a minute or so: a PR opened just now may be missing. If I name a PR or expect more
than you see, also list by author (`--author <login> --search '"Add" in:title'`).

Pair them via the `Related PR:` link in the main PR. If the body has several (e.g. a struck-through `~Related PR:
…~` and a newer one), use the last one that isn't struck through, and check that PR is open. An infra PR that no
main PR links to, or a main PR without a valid link, gets reported, not processed.

**Waiting on the contributor**: a pair whose latest maintainer comment asks the contributor for changes, with no
commit or comment from the contributor since, isn't grouped or re-reviewed. List it in the final report (see
"Stale PRs").

**Batches and units**: group pairs into batches by committer and library. When one person has opened several pairs
for the same library (e.g. `zap` v1.25.0 … v1.28.0), those form one batch; their `sys` and `go_cmp` pairs are
separate batches. Join batches into one unit when one needs another (step 2, "Dependencies"): merge the dependency
first, and test the dependents against the version it adds. If the dependency is blocked, so are the dependents.
Then decide what to process (see "Decide what to process").

In the step 4 question for a unit with more than one pair, ask which merge strategy to use (step 5):

- **Sequential**: merge each pair in turn, in ascending version order per library, resolving the conflicts each
  merge causes in the rest.
- **Combined branch**: merge all of the unit's PR heads into one new branch per repo on origin, open one PR
  per repo from those branches, merge them, and close the originals.

## 2. Correctness review

First read both PRs' conversations: `gh pr view <N> -R <repo> --json comments,reviews` and the inline review
comments (`gh api repos/<repo>/pulls/<N>/comments`). A maintainer may already have asked for changes, or the
contributor may have explained something (e.g. why a separate release repo is used). Don't contradict an earlier
decision or ask the contributor the same thing twice.

Then read both diffs in full (`gh pr diff <N> -R <repo>`), the library's existing entries, if any, in
`libraries.yaml` and the properties file on main, and a couple of neighbouring libraries of the same language/type
for the conventions. For a large batch, review every pair's diff; the entries differ only by version. Record each
problem as **fixable** (you will fix it in step 5) or **blocking** (it needs the contributor or me).

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
- **Windows** (C++ only): MSVC compilers read libraries from `etc/config/c++.amazonwin.properties`, not from the
  Linux file, and the wizard doesn't touch it. Windows servers have no `/opt/compiler-explorer/libs`, so every entry
  there is `packagedheaders=true` with no `.path=`: CE downloads the library's conan header package and adds
  `-I<dir>/<id>/include`. Don't add Windows entries in the wizard PR: the proper way is infra's
  `bin/ce_install cpp-library generate-windows-props --input-file <cemain>/etc/config/c++.amazonwin.properties
  --output-file <same>`, which copies the Linux libraries into the Windows file. It reads the Linux libraries from
  cemain `main`, so run it in its own PR after the wizard PRs merge, and review the whole diff: it also picks up
  every other library that drifted since the last sync. Hand-write a block only when I say a library is a special
  case. Either way, new Windows entries need a Windows build after merge (step 6).

### CI

`gh pr checks <N> -R <repo>` on both PRs. Failing CI is blocking unless one of your fixes is meant to address it.
Pending CI is not a finding: carry on, and wait for it to go green before merging (step 5). Green CI is not
evidence of correctness either: infra CI doesn't install or build the library.

## 3. Local verification

This is the pre-merge gate. It is only needed while the wizard's output is in doubt (see "Track record" below).

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

### Track record (temporary)

The local verification exists because the wizard's correctness is still unproven. Once it's proven for a kind of
library, the local build can become report-only for that kind. Decide this per **category**: language plus the
yaml `build_type` and `lib_type` exactly as written (missing counts as its own value), e.g. `c++ / cmake / static`,
`go / gomod`, `c++ / none`.

1. List merged wizard pairs, newest first: `gh pr list -R compiler-explorer/compiler-explorer --search
   'ce-lib-wizard in:body is:merged' --limit 200 --json number,title,mergedAt,body --jq 'sort_by(.mergedAt) |
   reverse | …'`, keeping only bodies with the wizard footer (the search also matches PRs that merely mention it).
   Get each one's category from the library's entry in the current `libraries.yaml`. If the category has fewer
   than 5 merged pairs, stop here: the local build is required.
2. For the 5 most recent pairs in the category, a pair is **clean** only if:
   - both PRs have exactly one non-merge commit; more means somebody fixed the wizard's output. Merges of `main`
     into the branch don't count:
     `gh api repos/<repo>/pulls/<N>/commits --jq '[.[] | select(.parents | length == 1)] | length'`;
   - nothing touched that library in the 30 days after the merge, other than the pair's own squash commits:
     `git log --since=<mergedAt> --until=<mergedAt+30d> -G'libs\.<id>\.' -- etc/config/<lang>.amazon.properties`
     in cemain, and `git log … -G'<name>' -- bin/yaml/libraries.yaml` in infra (check hits by eye). Use `-G`, not
     `-S`: `-S` misses edits inside an existing block.
3. If all 5 are clean, and step 2 found nothing in this pair, you may skip the local build: say so in the report
   ("skipped: 5/5 recent <category> pairs clean"). Otherwise run it.

At the time of writing, no category qualifies (about half of all wizard PRs were closed unmerged, and most merged
ones needed fixup commits). When a category qualifies, say so in the final report; I'll decide whether to relax
step 3.

## 4. Report and ask

For each unit, report:
- libraries, language, versions, author, and both PR links per pair
- every finding from step 2, marked fixable or blocking, and what you'll change for each fixable one
- the wizard bugs behind them: the issue you'll file (or the existing one) and the fix you'll publish
- the local verification result (or why it was skipped, per "Track record")
- whether it needs a real library build, and with which compiler

Then ask once (AskUserQuestion): **proceed with this unit?** For a unit with more than one pair, offer the two merge
strategies from step 1 as options. Recommend combined for 3 or more pairs, or when maintainer edits are disabled on
any of them. Otherwise recommend sequential. A yes covers everything from here to the end for that unit: filing
wizard issues and publishing their fixes, fixups, merges, the build, the comments and closing superseded PRs. It
does not carry over to the next unit.

**Blocking findings** take out only the pairs that depend on them (later versions of the same library, and the
batches that need it). Ask to proceed with the rest of the unit, listing what was taken out and why. For the
pairs taken out, offer to post a review comment to the contributor describing the problems; post it only after I
say yes. If nothing is left to proceed with, only make that offer.

## 5. Fix, then merge infra and main

Do all of this in worktrees under `$SCRATCH`, never in my checkouts. For cemain, create one the same way as for
infra in step 3 (`$SCRATCH/cemain-<N>`); if its pre-commit hook fails for lack of `node_modules`, run `npm ci` there.
Commit only the files you changed (`git add <file>`, then a plain `git commit`). Never force push, never amend,
never rewrite the contributor's history.

**Fixups**: if `maintainerCanModify` is true, push a normal commit on top of the contributor's branch, from the
PR's worktree: `git push https://github.com/<owner>/<repo>.git HEAD:<headRefName>`, with `<owner>` and `<repo>`
from step 1's `headRepositoryOwner.login` and `headRepository.name`. If maintainer edits are disabled, use the
combined-branch strategy (even for a single pair), or offer to ask the contributor for the fix. Fixes for a wizard
bug use the fix agent's derived output ("Rule for wizard bugs"), so wait for it before pushing them.

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

**Unit, combined branch**: both repos only allow squash merges, so the originals can't be marked as merged. One
combined PR per repo replaces them. This doesn't need maintainer edits and never pushes to the contributor's fork.
Do it in each repo:
1. `git -C /opt/compiler-explorer/<repo> worktree add -b lib-wizard/<author>-<lang>-<libs>
   $SCRATCH/<repo>-combined origin/main`, e.g. `lib-wizard/jdoe-go-zap`.
2. For each original PR in ascending version order (dependencies first): `git merge --no-ff lib-wizard-<N>` (the
   ref fetched in step 3). Resolve each conflict into a single block per library with the versions appended
   (step 2, "Existing library").
3. Add the step 2 fixups as separate commits.
4. Push the branch to origin (not a fork) and open the PR. In the body, list every original PR (`Supersedes #a,
   #b, …`), and give the cemain PR a `Related PR:` link to the combined infra PR. Credit the contributor with a
   `Co-authored-by: <name> <email>` line, taking the name and email from their wizard commit:
   `gh api repos/<repo>/pulls/<N>/commits --jq '.[0].commit.author | "\(.name) <\(.email)>"'`.
5. Wait for green CI, then merge with the credit passed explicitly. infra's squash message is blank by default, so
   the PR body alone doesn't reach the commit:
   `gh pr merge <N> -R <repo> --squash --body "Supersedes #a, #b, …"$'\n\n'"Co-authored-by: <name> <email>"`.
6. Leave the originals open; they're closed in step 7, after the build.

## 6. Real library build

Skip the Linux build for anything that doesn't need building (header-only, or libraries whose yaml says
`build_type: none` or `never`); the Windows build at the end of this step still applies. C libraries can't be
built this way: `lin-lib-build.yaml` has no C branch and would treat `libraries/c/<name>` as a C++ library name.
Skip them and say so in the final report.

Otherwise, after both merges (for a unit, after the last or combined ones), run the Linux library build once per
library, and for **one** compiler only. This is a smoke test of the production build, not a full rollout:
`scripts/dispatch-lib-build.sh lin libraries/<lang>/<name> <compiler-id>`. It dispatches the build, finds the new
run among your own dispatches (every run has the same title and its inputs aren't shown), waits for it, and prints
the `Detected language: …, library: …` line that confirms it was yours plus the build summary. It exits non-zero
unless at least one package built OK. Run one build at a time.

Pick a current GCC or Clang release id (C++), or the latest release of the language's own compiler (e.g. a
`gl…` id for Go). Look the id up in `etc/config/<lang>.amazon.properties`, not from memory. It doesn't have to be
installed locally.

Wait for the run to finish. If it fails: download the failure artifact (`gh run download <run-id> -R
compiler-explorer/infra`), diagnose, and report it. Both sides are already merged, so tell me what a follow-up fix
or revert would look like. Don't open one without asking.

**Windows** (only if the library was added to `c++.amazonwin.properties`, step 2): this is needed even for
header-only libraries, because Windows serves them as conan packages. Run `scripts/dispatch-lib-build.sh win <name>
<compiler-id>`, with the bare library name (no `libraries/c++/` prefix). For header-only libraries the builder always
uses `mingw64_ucrt_gcc_1220`, whatever compiler you pass. **The run's status means nothing**: it reports success
when the build failed, because the script drops ce_install's exit status. Go by the script's exit code, which
checks the log for `N packages built OK`. Header-only packaging runs the library's own CMake install; when that
installs no headers (no `CMakeLists.txt`, or no install rules), the builder copies the repo's `include/`, or else
its root, into the package instead.

## 7. Comment

Post one short comment on the main PR (and a one-liner on the infra PR pointing to it). Don't sign it, and don't
add any attribution line. Say:
- thanks
- what you changed, if anything (briefly: "set the Go import_path", "merged versions into the existing block")
- that it goes live with the next site release, and for built libraries, that only one compiler was built so far
  and other compilers get it from the regular library builds
- the minimum `-std=` (or Go/Rust toolchain version) if step 2 found one

For sequential units, one comment on the last main PR of each library, covering all its versions, is enough. For
combined units, close each original now with that comment, starting `Included in #<combined>, thanks!`
(`gh pr close <N> -R <repo> --comment …`).

## 8. Final report

Per unit: merged / skipped / blocked, fixups made, local verification result, build run URL and result, and anything
that needs my attention:
- failed builds after merge, and C libraries that couldn't be built
- blocked PRs, with or without a contributor comment posted
- the wizard issues filed and their fix PRs
- track-record categories that now qualify
- the Go dry-run conan post bug while it's open
- the units left open for a later run

Deployment is out of scope: libraries go out with the next normal release.

**Stale PRs**: list the pairs waiting on the contributor (step 1). For any where our comment has had no reply for 30
days, offer to close it with a short comment saying it can be reopened. Never close one without asking. List
blocked pairs where I declined to comment too, so I can decide again.

Last, remove the worktrees (`git -C /opt/compiler-explorer/<repo> worktree remove --force <path>`), delete the
`lib-wizard-<N>` branches and the combined branches' local copies, and delete `$SCRATCH`.
