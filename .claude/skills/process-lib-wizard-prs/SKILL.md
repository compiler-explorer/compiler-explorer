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

The steps are split over files in this skill's directory (paths in this skill are relative to it). Read each file
when this one sends you there, **before** doing the step it covers, even if you think you know what it says:
- `review-and-merge.md`: steps 2–5, review through merge
- `separate-agents.md`: only when I choose to process units in separate agents
- `windows.md`: C++ libraries only
- `wizard-bugs.md`, `combined-branch.md`, `track-record.md`: when `review-and-merge.md` points to them

## Decide what to process

Run step 1 first to find and group the pairs. If there is more than one unit, stop before step 2 and show me the
units (committer, libraries, versions, PR numbers, and why batches were joined into a unit). Then ask
(AskUserQuestion) which to do:

- **One unit first**: pick it, process it to the end, then come back to this question for the rest.
- **All at once, in separate agents**: one background agent per unit. Read `separate-agents.md` first.

Everything doesn't have to be done in one go. Units I don't pick are left open for a later run; just list them in
the final report.

## 1. Find the pairs

Run `scripts/list-wizard-prs.sh`. It lists the open wizard PRs in both repos, one line each, with the fields the
later steps need and the main PR's `Related PR:` link.

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

## 2–5. Review, verify, ask, fix and merge

Read `review-and-merge.md` and follow it for the unit. Come back here for step 6 once both sides are merged.

## 6. Real library build

Skip the Linux build for anything that doesn't need building (header-only, or libraries whose yaml says
`build_type: none` or `never`); a Windows build (`windows.md`) still applies. C libraries can't be built this way:
`lin-lib-build.yaml` has no C branch and would treat `libraries/c/<name>` as a C++ library name. Skip them and say
so in the final report.

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
