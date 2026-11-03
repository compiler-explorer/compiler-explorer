# Track record (temporary)

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
