#!/usr/bin/env python3
"""CLI for GitHub automation tools for Compiler Explorer."""

import click

from gh_tool.ai_duplicate_analyzer import filter_groups_with_ai
from gh_tool.duplicate_finder import fetch_issues, find_duplicates, generate_report


@click.group()
def main():
    """GitHub automation tools for Compiler Explorer."""
    pass


@main.command("find-duplicates")
@click.argument("output_file", type=click.Path())
@click.option(
    "--threshold",
    default=0.6,
    type=click.FloatRange(0, 1),
    help="Similarity threshold between 0 and 1 (default: 0.6)",
)
@click.option(
    "--state",
    type=click.Choice(["all", "open", "closed"]),
    default="open",
    help="Issue state to check (default: open)",
)
@click.option(
    "--min-age",
    default=0,
    type=int,
    help="Only check issues older than N days (default: 0)",
)
@click.option(
    "--limit",
    default=1000,
    type=int,
    help="Maximum number of issues to fetch (default: 1000)",
)
@click.option(
    "--repo",
    default="compiler-explorer/compiler-explorer",
    type=str,
    help="GitHub repository in owner/repo format (default: compiler-explorer/compiler-explorer)",
)
@click.option(
    "--use-ai",
    is_flag=True,
    help="Use AI to refine duplicate detection (requires ANTHROPIC_API_KEY in .env)",
)
@click.option(
    "--ai-confidence",
    default=0.7,
    type=click.FloatRange(0, 1),
    help="Minimum AI confidence score to keep a group (default: 0.7)",
)
def find_duplicates_cmd(output_file, threshold, state, min_age, limit, repo, use_ai, ai_confidence):
    """Find potential duplicate issues in the compiler-explorer repository.

    OUTPUT_FILE is the path where the markdown report will be saved.

    Examples:

        \b
        # Find duplicates in open issues
        gh_tool find-duplicates /tmp/duplicates.md

        \b
        # High confidence matches only
        gh_tool find-duplicates /tmp/report.md --threshold 0.85

        \b
        # Check all issues (including closed)
        gh_tool find-duplicates /tmp/all.md --state all
    """
    # Fetch issues
    click.echo(f"Fetching {state} issues...", err=True)
    issues = fetch_issues(state, limit, repo)

    # Find duplicates (broadphase)
    click.echo(f"Analyzing {len(issues)} issues with threshold {threshold}...", err=True)
    groups = find_duplicates(issues, threshold, min_age)

    # Refine with AI if requested
    if use_ai:
        groups = filter_groups_with_ai(groups, ai_confidence)

    # Generate report
    generate_report(groups, output_file)
    click.echo(f"Found {len(groups)} potential duplicate groups", err=True)


if __name__ == "__main__":
    main()
