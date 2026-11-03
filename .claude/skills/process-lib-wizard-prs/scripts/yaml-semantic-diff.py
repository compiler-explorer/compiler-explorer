#!/usr/bin/env python3
"""Check an infra wizard PR for collateral damage in bin/yaml/libraries.yaml.

Compares the PR head with its merge base (not base.sha: if main moved on since the wizard ran, main's newer changes
would show up as damage), after removing what the PR is meant to add. Key order, quoting and comments are ignored.

Usage: yaml-semantic-diff.py <infra-pr-number> <added> [<added> ...]
  <added> is a /-separated key path to a new library, e.g. libraries/c++/mylib, or <path>=<target> for a new
  target of an existing library, e.g. libraries/c++/mylib=1.2.3.
"""

import subprocess
import sys

import yaml

INFRA = "/opt/compiler-explorer/infra"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", INFRA, *args], text=True)


def load(ref: str):
    return yaml.safe_load(git("show", f"{ref}:bin/yaml/libraries.yaml"))


def target_name(target) -> str:
    return str(target.get("name") if isinstance(target, dict) else target)


def remove_added(doc, spec: str) -> None:
    path, _, target = spec.partition("=")
    *parents, key = path.split("/")
    node = doc
    for parent in parents:
        node = node[parent]
    if key not in node:
        sys.exit(f"{path} is not in the PR's libraries.yaml")
    if target:
        node[key]["targets"] = [t for t in node[key]["targets"] if target_name(t) != target]
    else:
        del node[key]


def differences(base, head, path: str = ""):
    if isinstance(base, dict) and isinstance(head, dict):
        for key in sorted(set(base) | set(head), key=str):
            if key not in base:
                yield f"added   {path}/{key}"
            elif key not in head:
                yield f"removed {path}/{key}"
            else:
                yield from differences(base[key], head[key], f"{path}/{key}")
    elif base != head:
        yield f"changed {path or '/'}"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    pr, added = sys.argv[1], sys.argv[2:]
    git("fetch", "-q", "origin", "main", f"+pull/{pr}/head:lib-wizard-{pr}")
    head_sha = git("rev-parse", f"lib-wizard-{pr}").strip()
    base_sha = git("merge-base", "origin/main", head_sha).strip()
    base, head = load(base_sha), load(head_sha)
    for spec in added:
        remove_added(head, spec)
    found = list(differences(base, head))
    print("identical" if not found else "SEMANTIC DIFFERENCE\n" + "\n".join(found))
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
