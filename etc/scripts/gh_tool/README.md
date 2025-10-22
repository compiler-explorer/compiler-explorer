# GitHub Tools for Compiler Explorer

CLI tools for automating GitHub repository management tasks.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for Python version and dependency management.

Install dependencies:

```bash
cd etc/scripts/gh_tool
uv sync
```

## Usage

Run from the gh_tool directory:

```bash
cd etc/scripts/gh_tool

# Get help
uv run gh_tool --help

# Get help for a specific command
uv run gh_tool find-duplicates --help
```

## Commands

### find-duplicates

Finds potential duplicate issues in the compiler-explorer repository using text similarity analysis (difflib.SequenceMatcher).

**Usage:**

```bash
# Basic usage (checks all open issues)
uv run gh_tool find-duplicates /tmp/duplicates-report.md

# Check all issues (including closed)
uv run gh_tool find-duplicates /tmp/all-duplicates.md --state all

# Adjust similarity threshold for higher confidence matches
uv run gh_tool find-duplicates /tmp/high-confidence.md --threshold 0.85

# Combine options
uv run gh_tool find-duplicates /tmp/report.md --threshold 0.7 --state all --min-age 30

# Use with a different repository
uv run gh_tool find-duplicates /tmp/other-repo.md --repo owner/repository
```

**Arguments:**

- `OUTPUT_FILE` (required) - Path to output markdown file

**Options:**

- `--threshold FLOAT` - Similarity threshold between 0 and 1 (default: 0.6)
  - 0.6 = 60% similar titles
  - Higher values = fewer, more confident matches
- `--state {all,open,closed}` - Which issues to check (default: open)
- `--min-age DAYS` - Only check issues older than N days (default: 0)
- `--limit INTEGER` - Maximum number of issues to fetch (default: 1000)
- `--repo TEXT` - GitHub repository in owner/repo format (default: compiler-explorer/compiler-explorer)

**Example Output:**

```markdown
# Potential Duplicate Issues

Found 5 potential duplicate groups:

## Group 1 (85% similar)
- #3201 [LIB REQUEST] numpy (12 comments, created 2021-03-15)
- #7778 [LIB REQUEST] numpy (0 comments, created 2024-01-10)

## Group 2 (72% similar)
- #4336 [COMPILER REQUEST]: Groovy (3 comments, created 2022-05-20)
- #6526 [COMPILER REQUEST]: Groovy (1 comments, created 2023-08-15)
```

**Performance:**

The duplicate detection algorithm uses O(n²) pairwise comparisons. For reference:
- ~850 issues: ~362,000 comparisons (~1-2 minutes)
- ~1,000 issues: ~500,000 comparisons (~2-3 minutes)

A progress bar shows real-time progress during the comparison phase.

**Requirements:**

- `gh` CLI must be installed and authenticated
- Read access to compiler-explorer/compiler-explorer repository

## Future Tools

This directory is intended to house additional GitHub automation scripts such as:

- Upstream project health checker (detect abandoned compiler/library projects)
- Label consistency validator
- Issue template compliance checker
- Automated triage reports

## Development

Run tests:

```bash
uv run pytest -v
```

Run linting:

```bash
uv run ruff check .
```

Format code:

```bash
uv run ruff format .
```
