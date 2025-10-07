"""Find potential duplicate issues in the compiler-explorer repository using text similarity."""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

import click


def fetch_issues(state: str = "open") -> list[dict[str, Any]]:
    """Fetch issues from GitHub using gh CLI."""
    print(f"Fetching {state} issues from GitHub...", file=sys.stderr)

    cmd = [
        "gh", "issue", "list",
        "--repo", "compiler-explorer/compiler-explorer",
        "--state", state,
        "--limit", "1000",
        "--json", "number,title,createdAt,updatedAt,state,labels,comments"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        issues = json.loads(result.stdout)
        print(f"Fetched {len(issues)} issues", file=sys.stderr)
        return issues
    except subprocess.CalledProcessError as e:
        print(f"Error fetching issues: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)


def calculate_similarity(title1: str, title2: str) -> float:
    """Calculate similarity ratio between two titles using SequenceMatcher."""
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()


def find_duplicates(
    issues: list[dict[str, Any]], threshold: float = 0.6, min_age_days: int = 0
) -> list[dict[str, Any]]:
    """Find potential duplicate issues based on title similarity."""
    # Filter issues by age if requested
    if min_age_days > 0:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=min_age_days)
        issues = [
            issue for issue in issues
            if datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00")) < cutoff_date
        ]
        print(f"Filtered to {len(issues)} issues older than {min_age_days} days", file=sys.stderr)

    # Find similar pairs
    duplicates = []
    seen = set()
    total_comparisons = len(issues) * (len(issues) - 1) // 2

    print(f"Comparing {len(issues)} issues ({total_comparisons:,} comparisons)...", file=sys.stderr)

    with click.progressbar(
        length=total_comparisons, label="Finding duplicates", file=sys.stderr
    ) as bar:
        for i, issue1 in enumerate(issues):
            for j, issue2 in enumerate(issues[i + 1 :], start=i + 1):
                bar.update(1)

                # Skip if already grouped
                pair = tuple(sorted([issue1["number"], issue2["number"]]))
                if pair in seen:
                    continue

                similarity = calculate_similarity(issue1["title"], issue2["title"])

                if similarity >= threshold:
                    duplicates.append({"similarity": similarity, "issues": [issue1, issue2]})
                    seen.add(pair)

    # Group similar issues together
    groups = []
    used = set()

    for dup in sorted(duplicates, key=lambda x: x["similarity"], reverse=True):
        issue1, issue2 = dup["issues"]
        num1, num2 = issue1["number"], issue2["number"]

        # Find existing group containing either issue
        found_group = None
        for group in groups:
            if num1 in [i["number"] for i in group["issues"]] or num2 in [i["number"] for i in group["issues"]]:
                found_group = group
                break

        if found_group:
            # Add to existing group if not already there
            for issue in [issue1, issue2]:
                if issue["number"] not in [i["number"] for i in found_group["issues"]]:
                    found_group["issues"].append(issue)
                    found_group["max_similarity"] = max(found_group["max_similarity"], dup["similarity"])
        else:
            # Create new group
            groups.append({
                "issues": [issue1, issue2],
                "max_similarity": dup["similarity"]
            })

    return groups


def format_issue(issue: dict[str, Any]) -> str:
    """Format issue for display."""
    labels = ", ".join(l["name"] for l in issue["labels"]) if issue["labels"] else "no labels"
    created = issue["createdAt"][:10]
    state = issue["state"]
    comment_count = len(issue.get("comments", []))

    return f"- #{issue['number']} {issue['title']} ({comment_count} comments, {state}, created {created})"


def generate_report(groups: list[dict[str, Any]], output_file: str) -> None:
    """Generate markdown report of duplicate groups."""
    output = []
    output.append("# Potential Duplicate Issues\n")

    if not groups:
        output.append("No potential duplicates found.\n")
    else:
        output.append(f"Found {len(groups)} potential duplicate groups:\n")

        for idx, group in enumerate(sorted(groups, key=lambda x: x["max_similarity"], reverse=True), 1):
            similarity_pct = int(group["max_similarity"] * 100)
            output.append(f"## Group {idx} ({similarity_pct}% similar)\n")

            for issue in sorted(group["issues"], key=lambda x: x["createdAt"]):
                output.append(format_issue(issue) + "\n")

            output.append("\n")

    report = "\n".join(output)

    with open(output_file, "w") as f:
        f.write(report)
    print(f"Report saved to {output_file}", file=sys.stderr)
