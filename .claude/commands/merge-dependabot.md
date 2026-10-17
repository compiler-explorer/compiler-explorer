# Merge and Deploy Dependabot PRs

Merge the open dependabot PRs in compiler-explorer/compiler-explorer, then deploy the result to staging and,
after my approval, to prod. Only dependabot PRs: if a PR's author is not `app/dependabot`, leave it alone.

Treat every dependabot PR as a security update that needs to ship. Don't skip the deploy because a change "looks
dev-only".

The one exception is a **scripts-only** PR: every changed file is under `etc/scripts/`. Those tools are
independent of CE itself, so a scripts-only PR gets the CI, changelog and local tool test gates, then is merged
with no local site run and no deploy. Check with the PR's `files` list. If even one file is outside
`etc/scripts/`, the PR goes through the full flow.

The only point where you stop and wait for me is before the prod deploy. Everything else runs end to end, but a
PR that fails a gate is skipped (not merged) and reported, never forced through.

## 1. Find the PRs

```bash
gh pr list --author app/dependabot --state open \
    --json number,title,headRefName,mergeStateStatus,files,url
```

If there are none, say so and stop.

## 2. Rebase via dependabot

For each PR whose `mergeStateStatus` is `BEHIND`, `DIRTY` or otherwise not up to date with main, comment
`@dependabot rebase` (`gh pr comment <N> --body '@dependabot rebase'`) and wait for dependabot to push the
rebased branch. Never rebase or push to a dependabot branch yourself.

After merging one PR, the others may fall behind. Ask for a rebase again before merging the next one.

## 3. Gates (all must pass per PR)

### CI green

`gh pr checks <N> --watch`. Every required check must pass. A failing check means skip the PR and report it.

### Changelog review

The PR body holds the release notes, changelog and commits dependabot pulled from upstream. Read them, and for
anything that's cut off, follow the links to the upstream release pages.

The most important checks, in order:

1. **Breaking changes**: anything labelled "breaking", "BREAKING CHANGE", "removed", "dropped support" or
   "no longer" anywhere in the range being skipped over, not only in the target release.
2. **Major version bumps**: any change in the major number (or the minor number for `0.x` packages). This
   includes transitive dependencies that move a major version in the lockfile diff.

Then use judgement about what CE actually uses from the package: grep for its imports and usage (`lib/`,
`static/`, `types/`, `etc/scripts/...`, workflow files) and decide whether a change affects us and whether I should
know about it, e.g. changed defaults, deprecated APIs we call, or behaviour changes in code paths we depend on.
Say nothing about changes that don't affect us.

A breaking or major change that affects code we use means: don't merge. Report it with what would have to change.
A breaking or major change that doesn't touch anything we use can be merged, but say in the report that you
checked and why it's safe.

### Local test run

Use a worktree so main stays clean:

```bash
git fetch origin <headRefName>
git worktree add ../dependabot-<N> origin/<headRefName>
```

Run the checks for whatever the PR touches:

- **npm** (`package.json`/`package-lock.json`): `npm ci`, `npm run ts-check`, `npm run lint-check`, `npm run test-min`
- **uv/pip under `etc/scripts/<tool>`**: `uv sync --locked` in that directory, then run the tool's own tests if
  it has any. Otherwise run the tool with `--help`, or the smallest invocation that exercises the bumped package
- **github-actions**: CI on the PR is the test, so nothing extra locally

### Local site run

Before anything goes near staging, run the site from the worktree the way production runs it. If it breaks here,
a staging deploy would only waste time and instances. Do this for every PR except scripts-only ones.

```bash
cp etc/config/*.local.properties ../dependabot-<N>/etc/config/
sed -i '/^staticUrl=/d' ../dependabot-<N>/etc/config/compiler-explorer.local.properties
cd ../dependabot-<N>
make run EXTRA_ARGS='--port 10241 --language c++'
```

My `compiler-explorer.local.properties` points `staticUrl` at the production CDN. Strip that line from the
worktree copy (never from the main checkout), or the page will request this build's bundles from the CDN, get
403s, and render a blank layout with no console errors.

`make run` does the production webpack and ts build, then starts `out/dist/app.js`. Run it in the background,
wait for the startup to finish (`Listening on http://localhost:10241/`), and check that it says
`serving static files from './out/webpack/static'`. Port 10241 avoids clashing with a dev instance I may have
running on 10240.

The startup log is noisy, and that noise isn't a failure: `Unable to stat ... compiler binary`, orphaned group
warnings and `Failed to create N out of M compilers` just mean those compilers aren't installed locally. A real
failure is a build error, a crash before `Listening on`, or stack traces from request handling.

Then check it:
- **Smoke compile**: the curl command from "Verify staging" below, against `http://localhost:10241/`.
- **Browser check** (claude-in-chrome): open http://localhost:10241 in a new tab, check the console for errors,
  and run a compile from the editor. Console capture only starts on the first `read_console_messages` call, so call
  it once before loading the page or reload after the first call. Otherwise load-time errors are missed. If the bumped package backs a visible feature, exercise that too.

Any build failure, startup error, console error or failed compile means skip the PR and report it. Stop the server
when you're done: kill the `node` process listening on 10241 (stopping the background `make` alone doesn't).

Remove the worktree afterwards with `git worktree remove --force ../dependabot-<N>`. `--force` is needed because
`.venv`/`node_modules` are untracked.

## 4. Merge

For each PR that passed every gate, one at a time:

```bash
gh pr merge <N> --squash
```

Never use `--admin` and never bypass branch protection. Request a fresh `@dependabot rebase` for the next PR if
it fell behind (see step 2).

If nothing was merged, stop here and report. If everything merged was scripts-only, also stop here and report.
There's nothing to deploy.

## 5. Find the build

Every commit to main gets a build tagged `gh-xxxx`. Wait for the main CI run of the last merge commit to finish,
then:

```bash
git fetch origin --tags
git tag --points-at origin/main | grep '^gh-'
```

Confirm the build was uploaded: `ce admin exec -- ce --env staging builds list`.

All `ce` commands run on the admin node through `ce admin exec -- ...`, using the local
`/opt/compiler-explorer/infra/bin/ce`. Running there lets the deploy do its HTTP health checks.

## 6. Staging

Run compiler discovery for the build and wait for it (about 5 minutes):

```bash
gh workflow run -R compiler-explorer/infra 'Compiler discovery' \
    -f environment=staging -f branch=main -f buildnumber=gh-xxxx
gh run watch -R compiler-explorer/infra <run-id>
```

Deploy (takes a while, so run it in the background and wait for it to finish):

```bash
ce admin exec -- ce --env staging blue-green deploy gh-xxxx --skip-confirmation
```

If it fails with a bounce lock, stop and tell me; don't bypass it. If it fails on a hash mismatch, stop and tell me
(the fix is a webpack hack-version bump, which is not a dependabot change).

### Verify staging

- **Smoke compile**: use the `compiler-explorer-staging` MCP if it's connected. Otherwise pick a current GCC id
  from `https://godbolt.org/staging/api/compilers/c++?fields=id,name` and check that `code` is 0 and real asm
  comes back. Without the `Accept` header the API returns text, not JSON:

  ```bash
  curl -s https://godbolt.org/staging/api/compiler/<id>/compile \
      -H 'Content-Type: application/json' -H 'Accept: application/json' \
      -d '{"source":"int square(int x){return x*x;}","options":{"userArguments":"-O2"}}'
  ```
- **Browser check** (claude-in-chrome): open https://godbolt.org/staging in a new tab, confirm the version via
  Other > Version Tree matches `gh-xxxx`, check the console for errors, and run a compile from the editor.
  If a bumped package backs a visible feature (e.g. a docenizer → asm docs tooltips), exercise that feature too.

If staging fails anything: don't go to prod. Run `ce admin exec -- ce --env staging blue-green rollback --skip-confirmation`,
work out which merged PR is likely responsible, and report back. Don't open revert PRs without asking me.

## 7. Stop for approval

Report:
- merged PRs (number, package, from → to, whether it was a security fix)
- skipped PRs and why
- relevant changelog notes (breaking or major changes and how they affect us)
- local test and staging verification results
- the build number

Then ask whether to deploy to prod. Don't continue without an explicit yes.

## 8. Prod

Run discovery for prod first, so `--skip-confirmation` doesn't refuse the deploy over missing discovery:

```bash
gh workflow run -R compiler-explorer/infra 'Compiler discovery' \
    -f environment=prod -f branch=main -f buildnumber=gh-xxxx
gh run watch -R compiler-explorer/infra <run-id>

ce admin exec -- ce --env prod blue-green deploy gh-xxxx --skip-confirmation --no-notify
```

Always pass `--no-notify`: dependabot PRs don't get "now live" comments.

Afterwards, run a smoke compile against https://godbolt.org and check `ce admin exec -- ce --env prod blue-green status`.
If prod misbehaves, roll back right away (`ce admin exec -- ce --env prod blue-green rollback --skip-confirmation`)
and tell me.

Once prod is live, clean up staging: `ce admin exec -- ce --env staging blue-green cleanup --skip-confirmation`.
Leave prod's old colour up for rollback, and remind me to run `ce --env prod blue-green cleanup` once it's
proven stable.
