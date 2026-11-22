#!/usr/bin/env bash
# Checks out a PR head in a worktree under the scratch directory, leaving the main checkout untouched.
# Usage: pr-worktree.sh <infra|cemain> <pr-number> <scratch-dir>
# Creates <scratch-dir>/<repo>-<pr-number> on local branch lib-wizard-<pr-number>. Re-running after a fixup
# refreshes the branch (the fetch uses +) but needs the old worktree removed first.
set -euo pipefail

repo=${1:?repo (infra or cemain)}
pr=${2:?PR number}
scratch=${3:?scratch directory}

case ${repo} in
infra | cemain) ;;
*)
    echo "unknown repo ${repo}" >&2
    exit 1
    ;;
esac

checkout=/opt/compiler-explorer/${repo}
worktree=${scratch}/${repo}-${pr}
git -C "${checkout}" fetch -q origin main "+pull/${pr}/head:lib-wizard-${pr}"
git -C "${checkout}" worktree add -q "${worktree}" "lib-wizard-${pr}"
# A fresh infra worktree has no python env, and ce_install refuses to run without one.
if [[ ${repo} == infra ]]; then
    make -C "${worktree}" ce >/dev/null
fi
echo "${worktree}"
