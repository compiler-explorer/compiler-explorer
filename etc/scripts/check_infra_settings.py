#!/usr/bin/env python3
"""Check whether a PR adds compilerType or C++ stdver values not present in
compiler-explorer/infra's init/settings.yml.

Only a curated list of language property files is examined — those whose
libraries are built via Conan in the infra repository.

Usage:
    ./etc/scripts/check_infra_settings.py [--base-ref origin/main]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

MARKER = "<!-- ce-infra-settings-check -->"

INFRA_SETTINGS_URL = "https://raw.githubusercontent.com/compiler-explorer/infra/main/init/settings.yml"

SCOPED_FILES = [
    "etc/config/c.amazon.properties",
    "etc/config/c.amazonwin.properties",
    "etc/config/c++.amazon.properties",
    "etc/config/c++.amazonwin.properties",
    "etc/config/cuda.amazon.properties",
    "etc/config/d.amazon.properties",
    "etc/config/fortran.amazon.properties",
    "etc/config/go.amazon.properties",
    "etc/config/rust.amazon.properties",
    "etc/config/sway.amazon.properties",
    "etc/config/zig.amazon.properties",
]

COMPILER_TYPE_RE = re.compile(r"^(?:group|compiler)\.\S+\.compilerType=([A-Za-z0-9_.+-]+)")
STDVER_RE = re.compile(r"-std=((?:c|gnu)\+\+[A-Za-z0-9_]+)")


def run_git_diff(base_ref: str, files: list[str]) -> str:
    cmd = ["git", "diff", f"{base_ref}...HEAD", "--", *files]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def collect_values(diff_text: str) -> tuple[set[str], set[str]]:
    """Return (compilerTypes, stdvers) that are net-new — present on '+' lines
    but absent from '-' lines of the same diff."""
    added_types: set[str] = set()
    removed_types: set[str] = set()
    added_stdvers: set[str] = set()
    removed_stdvers: set[str] = set()

    for line in diff_text.splitlines():
        if not line or line[0] not in "+-":
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        content = line[1:]
        ct = COMPILER_TYPE_RE.search(content)
        sv = STDVER_RE.findall(content)
        if line[0] == "+":
            if ct:
                added_types.add(ct.group(1))
            added_stdvers.update(sv)
        else:
            if ct:
                removed_types.add(ct.group(1))
            removed_stdvers.update(sv)

    return added_types - removed_types, added_stdvers - removed_stdvers


def fetch_infra_settings(url: str) -> tuple[set[str], set[str]]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = yaml.safe_load(resp.read())
    compiler_keys = set((data.get("compiler") or {}).keys())
    stdvers = {s for s in (data.get("stdver") or []) if s}
    return compiler_keys, stdvers


def format_comment(missing_types: set[str], missing_stdvers: set[str]) -> str:
    lines = [
        MARKER,
        "### Infra `settings.yml` may need updating",
        "",
        "This PR adds values that are not currently listed in "
        "[`compiler-explorer/infra/init/settings.yml`]"
        "(https://github.com/compiler-explorer/infra/blob/main/init/settings.yml):",
        "",
    ]
    if missing_types:
        lines.append("**New `compilerType` values:**")
        lines.extend(f"- [ ] `{t}`" for t in sorted(missing_types))
        lines.append("")
    if missing_stdvers:
        lines.append("**New `stdver` values:**")
        lines.extend(f"- [ ] `{s}`" for s in sorted(missing_stdvers))
        lines.append("")
    lines.append(
        "Please open a matching PR in "
        "[compiler-explorer/infra](https://github.com/compiler-explorer/infra) "
        "to add these to `init/settings.yml` — Conan library builds will fail "
        "otherwise."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main", help="Git ref to compare against")
    parser.add_argument("--infra-settings-url", default=INFRA_SETTINGS_URL)
    args = parser.parse_args()

    existing = [f for f in SCOPED_FILES if Path(f).exists()]
    if not existing:
        print("No scoped property files found in working tree; nothing to check.")
        return 0

    diff_text = run_git_diff(args.base_ref, existing)
    if not diff_text.strip():
        print("No changes to scoped property files.")
        return 0

    added_types, added_stdvers = collect_values(diff_text)
    if not added_types and not added_stdvers:
        print("No new compilerType or stdver values detected in additions.")
        return 0

    known_types, known_stdvers = fetch_infra_settings(args.infra_settings_url)
    missing_types = added_types - known_types
    missing_stdvers = added_stdvers - known_stdvers

    if not missing_types and not missing_stdvers:
        print("All added compilerType/stdver values are already present in infra/init/settings.yml.")
        return 0

    print(format_comment(missing_types, missing_stdvers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
