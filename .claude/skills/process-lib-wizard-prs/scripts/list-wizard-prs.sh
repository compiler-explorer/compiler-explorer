#!/usr/bin/env bash
# Lists the open PRs made by ce-lib-wizard in both repos, one line each:
#   <number> <author> <title> <owner>/<repo>:<branch> mcm=<maintainerCanModify> <mergeStateStatus> <createdAt> <related>
# <related> is the last `Related PR:` link that isn't struck through (main PRs only).
# The search also matches PRs that merely mention the wizard, so only bodies with its footer are kept.
set -euo pipefail

fields=number,title,author,body,headRefName,headRepository,headRepositoryOwner,maintainerCanModify,mergeStateStatus,createdAt
for repo in compiler-explorer infra; do
    echo "== compiler-explorer/${repo}"
    gh pr list -R "compiler-explorer/${repo}" --search 'ce-lib-wizard in:body is:open' --limit 200 --json "${fields}" \
        --jq '.[] | select(.body | test("PR created with \\[ce-lib-wizard"))
            | [.number, .author.login, .title,
               "\(.headRepositoryOwner.login)/\(.headRepository.name):\(.headRefName)",
               "mcm=\(.maintainerCanModify)", .mergeStateStatus, .createdAt,
               ([.body | scan("(?m)^[^~\\n]*Related PR: (https://\\S+)")] | last | if . then .[0] else "-" end)]
            | map(tostring) | join("\t")'
done
