#!/usr/bin/env bash
# Dispatches one library build in infra, waits for it, and reports the result from the log.
# Usage: dispatch-lib-build.sh lin <libraries/<lang>/<name>> <compiler-id>
#        dispatch-lib-build.sh win <name> <compiler-id>
# Every run has the same title and its inputs aren't shown, so the run is found as the one new id among your own
# dispatches: run one build at a time. Exits non-zero unless the log reports at least one package built OK, because
# the run's own status isn't reliable (Windows runs report success when the build failed).
set -euo pipefail

platform=${1:?lin or win}
library=${2:?library}
compiler=${3:?compiler id}

case ${platform} in
lin)
    workflow=lin-lib-build.yaml
    extra=(-f branch=main)
    ;;
win)
    workflow=win-lib-build.yaml
    extra=()
    ;;
*)
    echo "unknown platform ${platform}" >&2
    exit 1
    ;;
esac

repo="compiler-explorer/infra"
me=$(gh api user --jq .login)
runs() {
    gh run list -R "${repo}" --workflow "${workflow}" --event workflow_dispatch --user "${me}" --limit 20 \
        --json databaseId --jq '.[].databaseId' | sort
}

before=$(runs)
gh workflow run "${workflow}" -R "${repo}" -f "library=${library}" -f "compiler=${compiler}" "${extra[@]}"

run=""
for _ in $(seq 30); do
    run=$(comm -13 <(echo "${before}") <(runs))
    if [[ -n ${run} ]]; then
        break
    fi
    sleep 3
done
if [[ -z ${run} || $(wc -l <<<"${run}") -ne 1 ]]; then
    echo "expected exactly one new run, found: ${run:-none}" >&2
    exit 1
fi

echo "run: https://github.com/${repo}/actions/runs/${run}"
gh run watch "${run}" -R "${repo}" >/dev/null || true
log=$(gh run view "${run}" -R "${repo}" --log)
grep -m1 'Detected language' <<<"${log}" || true
grep -E 'packages built|failed to build' <<<"${log}" || true
grep -qE '[1-9][0-9]* packages built OK' <<<"${log}"
