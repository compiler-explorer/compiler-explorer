# Processing units in separate agents

One background agent per unit (Agent tool):
- Before spawning, fetch every chosen PR head once yourself
  (`git -C /opt/compiler-explorer/<repo> fetch -q origin main +pull/<N>/head:lib-wizard-<N>` per PR), so agents
  don't all fetch in my checkouts at the same moment. If an agent's later fetch fails with `cannot lock ref`, it
  retries.
- Brief each agent with the path of `review-and-merge.md` (not `SKILL.md`: it doesn't need the rest), its unit's
  PR numbers and their `scripts/list-wizard-prs.sh` lines, a unit name (e.g. `jdoe-zap`), the batch and unit terms
  from `SKILL.md`, and these rules. It runs steps 2–3 and returns the step 4 report **instead of asking**:
  subagents can't ask me. You ask me per unit, as step 4 describes, and send the answer back to that agent
  (SendMessage) so it carries on with step 5.
- The agent stops after its merges and returns what it changed. You run steps 6–8 for that unit yourself, one unit
  at a time, so only one build dispatch is in flight while you look for its run id.
- Each agent works only in its own scratch directory, `$SCRATCH` = `<session scratchpad>/lib-wizard-<unit>`, with
  its own worktrees of **both** repos there (step 3). My checkouts are never switched or committed to.
- Wizard bugs are handled by you, not the unit agents: they report defects, you read `wizard-bugs.md` and apply its
  rule once per bug (one fix agent, one issue), and forward the derived output to every unit it affects.
