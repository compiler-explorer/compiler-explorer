# Wizard bugs

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
