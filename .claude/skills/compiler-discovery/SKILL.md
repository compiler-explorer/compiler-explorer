---
name: compiler-discovery
description: How to use Compiler Explorer's compiler discovery (--discovery-only, --prediscovered) locally. Use when someone wants a local CE server (dev or their own install) to start fast or keep a fixed set of compilers across restarts and test runs, when asked to "test with discovery" or check a change on an instance started from a discovery file versus one that discovers its own compilers, when a change touches CompilerInfo, compiler-finder, tools or libraries, or when a bug only shows on prod/staging instances.
---

# Compiler discovery

`--discovery-only <file>` finds the compilers, writes them to a JSON file and exits. `--prediscovered <file>` starts
from that file instead of probing every compiler. Prod and staging always start from a file made on a runner by the
infra "Compiler discovery" workflow; a dev instance never does unless you ask it to.

## Running a local server from a fixed set of compilers

Discover once, then start from the file on every run: startup is faster, and the compiler list stays the same
across restarts and test runs.

1. Discover with the config the server normally uses (`etc` by default), into a file git doesn't track:
   `.claude/skills/compiler-discovery/scripts/discover.sh etc <dir>/discovered.json c++`
2. Optionally keep only the compilers you want:
   `jq '[.[] | select(.id as $id | $ARGS.positional | index($id))]' <dir>/discovered.json --args g161 clang2110 > <dir>/fixed.json`
   (`--args` must come after the input file.)
3. Start from it with the same `--language` list:
   - dev mode: `make dev EXTRA_ARGS='--prediscovered <dir>/fixed.json --language c++'`
   - a built install: add `--prediscovered <dir>/fixed.json` to its usual arguments

By default discovery leaves out any compiler that fails to load, so a broken one just goes missing from the file.
The prod workflow passes `--exit-on-compiler-failure` so that any such failure fails the run; `discover.sh` takes it
as its first argument and then lists the failed compilers. Use it when the config should only list compilers that
are installed, like your own server's. With CE's full config on a dev machine it always fails, because most of those
compilers aren't installed there.

The file freezes the compiler list and every compiler and group property. After editing those or installing a
compiler, discover again. Language-level settings, libraries and tools are still read from config.

## Testing a change in both modes

On CE's own machines `etc/config/*.local.properties` are prod copies that write to the prod S3 cache and stats, so
test from a throwaway root:

1. `.claude/skills/compiler-discovery/scripts/make-test-root.sh <scratch>/etc`: a copy of `etc/config` with local
   in-memory caches and cache-access logging on.
2. `.claude/skills/compiler-discovery/scripts/discover.sh <scratch>/etc <scratch>/disc.json c++ c rust`
3. Start one instance from the file and one that discovers live, on free ports. `start.sh` returns once the instance
   is up, or fails with the end of its log:
   - `.claude/skills/compiler-discovery/scripts/start.sh <scratch>/etc 10250 <scratch>/pre.log <scratch>/disc.json c++ c rust`
   - `.claude/skills/compiler-discovery/scripts/start.sh <scratch>/etc 10251 <scratch>/live.log - c++ c rust`
4. `python3 -I .claude/skills/compiler-discovery/scripts/compare-servers.py http://localhost:10250 http://localhost:10251`
   sends the same compilations to both and reports differences. Edit its `CASES` to cover what the change touches.
5. To compare cache keys, send identical requests without `bypassCache` and compare the
   `Compilation cache (get|put) <hash>` lines in the two logs; they must match.
6. Stop them: `pkill -f -- '--port 1025[01]'`

Only differences between the two instances point at discovery. Failures on both are the environment (a missing
library or tool, a bad flag). The live log also has errors for compilers not installed on this machine; that's
expected.
