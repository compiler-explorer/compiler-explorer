# Unit, combined branch

Both repos only allow squash merges, so the originals can't be marked as merged. One combined PR per repo replaces
them. This doesn't need maintainer edits and never pushes to the contributor's fork. Do it in each repo:

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
5. Wait for green CI, check `mergeStateStatus` as in step 5, then merge with the credit passed explicitly. infra's
   squash message is blank by default, so the PR body alone doesn't reach the commit:
   `gh pr merge <N> -R <repo> --squash --body "Supersedes #a, #b, …"$'\n\n'"Co-authored-by: <name> <email>"`.
   Never use `--admin`.
6. Leave the originals open; they're closed in step 7, after the build.
