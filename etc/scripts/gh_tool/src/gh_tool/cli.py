#!/usr/bin/env python3
"""CLI for GitHub automation tools for Compiler Explorer."""

import sys

import click

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
    type=float,
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
def find_duplicates_cmd(output_file, threshold, state, min_age):
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
    # Validate threshold
    if not (0 <= threshold <= 1):
        click.echo("Error: threshold must be between 0 and 1", err=True)
        sys.exit(1)

    # Fetch issues
    click.echo(f"Fetching {state} issues...", err=True)
    issues = fetch_issues(state)

    # Find duplicates
    click.echo(f"Analyzing {len(issues)} issues with threshold {threshold}...", err=True)
    groups = find_duplicates(issues, threshold, min_age)

    # Generate report
    generate_report(groups, output_file)
    click.echo(f"Found {len(groups)} potential duplicate groups", err=True)


if __name__ == "__main__":
    main()
