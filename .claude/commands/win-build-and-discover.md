# Windows Build and Discovery

Build the Windows version of CE for a main commit and run Windows compiler discovery for it. The commit is
`$ARGUMENTS` if given, otherwise the latest commit on `origin/main`.

The steps run in order, each only after the one before it succeeded. If a step fails, stop and report it. Don't
retry it or start the next one.

## 1. Find the commit and its build number

```bash
git fetch -q origin main && git fetch -q --tags origin
sha=$(git rev-parse origin/main)   # or the commit from $ARGUMENTS
git tag --points-at $sha           # the build number, e.g. gh-19854
```

The `gh-NNNNN` tag is created by the main CI pipeline. Don't wait for the rest of CI to finish: as soon as the
tag exists, go on. If the commit has no tag yet, poll in a background command (fetch tags once a minute) until
it appears. If the "Compiler Explorer" run on the commit fails before tagging, stop and report it.

## 2. Start the CE Windows Build

```bash
gh workflow run deploy-win.yml -f buildnumber=<gh-NNNNN>
```

Its run is titled `Build CE Windows version <gh-NNNNN>`. Find it with
`gh run list --workflow deploy-win.yml -L 5 --json databaseId,displayTitle` and watch it in the background with
`gh run watch <id> --exit-status`.

## 3. Start Windows compiler discovery

This workflow lives in the infra repo. Start it only after the Windows build succeeded, because it needs the
Windows zip for this build number:

```bash
gh workflow run win-compiler-discovery.yml -R compiler-explorer/infra \
    -f environment=winstaging -f branch=main -f buildnumber=<gh-NNNNN>
```

Use `winstaging` unless I ask for `winprod`. Watch the run in the background with
`gh run watch <id> -R compiler-explorer/infra --exit-status`, and tell me when it finishes, whether it passed or
failed. If it failed, include the failing step's log tail.

## 4. Report and offer the follow-ups

Report the commit, the build number, and the links to the Windows build run and the discovery run.

The usual next step is deploying this build to Windows staging, a test environment, with blue-green (see
`docs/release_new_version.md` and `docs/blue_green_deployment_strategy.md` in infra):

```bash
ce admin exec -- ce --env winstaging blue-green status
ce admin exec -- ce --env winstaging blue-green deploy --capacity 1 --skip-confirmation <gh-NNNNN>
```

Run `ce` as `bin/ce` from the infra checkout, and the blue-green commands through `ce admin exec` so the compiler
registration check runs from the admin node. Before deploying:

- If the inactive colour still has instances from an earlier deploy, the deploy refuses. Scale it down first with
  `ce admin exec -- ce --env winstaging blue-green cleanup --skip-confirmation`.
- If the active colour has no instances, `deploy` would match that and start none, hence `--capacity 1`.

Keep the deploy's full output in a file: the error that matters is above the Python traceback.

Deploying to `winprod` comes after testing on winstaging, and only when I ask for it:

- Run discovery again with `-f environment=winprod`. Each run uploads to one environment only.
- Claude Code's permission check refuses production deploys, so give me the command to run myself with `!`:
  `cd /opt/compiler-explorer/infra && bin/ce admin exec -- ce --env winprod blue-green deploy --skip-confirmation <gh-NNNNN>`.
  winprod has instances, so it needs no `--capacity`.
- Afterwards the old colour keeps running for a rollback (`blue-green rollback`) until it is scaled down with
  `blue-green cleanup`. Clean it up, and shut winstaging down with `blue-green shutdown`, only when I ask.
