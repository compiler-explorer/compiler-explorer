"""AI-powered duplicate issue detection using Claude."""

import os
from typing import Any

import click
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_anthropic_client() -> Anthropic | None:
    """Get Anthropic client if API key is available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def analyze_duplicate_group(issues: list[dict[str, Any]], client: Anthropic) -> dict[str, Any]:
    """Use AI to determine if a group of issues are truly duplicates.

    Args:
        issues: List of issue dictionaries with at least 'number' and 'title' fields
        client: Anthropic client instance

    Returns:
        dict with:
            - is_duplicate: bool - whether issues are duplicates
            - confidence: float - confidence score 0-1
            - reasoning: str - explanation of the decision
    """
    # Format issues for the prompt
    issue_list = "\n".join(f"- Issue #{issue['number']}: {issue['title']}" for issue in issues)

    prompt = f"""You are analyzing GitHub issues to determine if they are EXACT duplicates.

Issues to analyze:
{issue_list}

STRICT RULES for duplicates:
1. Different named tools/libraries are DIFFERENT: "fasm" ≠ "YASM" ≠ "AsmX" (even if all assemblers)
2. Similar categories are NOT duplicates: requesting different compilers/libraries/languages is NOT a duplicate
3. Related features are NOT duplicates: "language tooltips" ≠ "language detection" (different features)
4. Only spelling/capitalization variants are duplicates: "NumPy" = "numpy", "Forth" = "FORTH"

EXAMPLES:
✓ DUPLICATE: "Add NumPy" + "Add numpy" (same library, different case)
✓ DUPLICATE: "GCC 13" + "GCC 13.1" (same compiler, version variants)
✗ NOT DUPLICATE: "fasm" + "YASM" (different assemblers, even though both are assemblers)
✗ NOT DUPLICATE: "OpenBLAS" + "OpenSSL" (different libraries starting with "Open")
✗ NOT DUPLICATE: "ARM execution" + "EWARM execution" (EWARM is specific toolchain, not general ARM)
✗ NOT DUPLICATE: "language tooltips" + "language detection" (different features in same domain)

Be VERY strict. Only mark as duplicate if they request the EXACT SAME thing with minor spelling/version variations.

Respond in JSON format:
{{
    "is_duplicate": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the response
        content = response.content[0].text.strip()

        # Debug logging for empty responses
        if not content:
            click.echo(f"    Warning: Empty response from AI for group with {len(issues)} issues", err=True)
            return {"is_duplicate": False, "confidence": 0.0, "reasoning": "Empty AI response"}

        # Try to extract JSON from the response
        import json

        # Find JSON block if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        return {
            "is_duplicate": result.get("is_duplicate", False),
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
        }

    except json.JSONDecodeError as e:
        click.echo(f"    Warning: Invalid JSON from AI (group size: {len(issues)})", err=True)
        click.echo(f"    First 200 chars of response: {content[:200] if content else '(empty)'}...", err=True)
        return {"is_duplicate": False, "confidence": 0.0, "reasoning": f"JSON parse error: {e}"}
    except Exception as e:
        click.echo(f"    Warning: AI analysis failed: {e}", err=True)
        # On error, assume it's not a duplicate to avoid false positives
        return {"is_duplicate": False, "confidence": 0.0, "reasoning": f"Error: {e}"}


def filter_groups_with_ai(groups: list[dict[str, Any]], min_confidence: float = 0.7) -> list[dict[str, Any]]:
    """Filter duplicate groups using AI analysis.

    Args:
        groups: List of duplicate groups from broadphase detection
        min_confidence: Minimum confidence score to keep a group (default 0.7)

    Returns:
        Filtered list of groups that AI confirms are duplicates
    """
    client = get_anthropic_client()
    if not client:
        click.echo("Warning: ANTHROPIC_API_KEY not found in environment or .env file", err=True)
        click.echo("Skipping AI analysis, returning all groups", err=True)
        return groups

    click.echo(f"Analyzing {len(groups)} groups with AI...", err=True)

    filtered_groups = []
    for idx, group in enumerate(groups, 1):
        num_issues = len(group["issues"])
        click.echo(f"  Analyzing group {idx}/{len(groups)} ({num_issues} issues)...", err=True)

        # Warn about very large groups
        if num_issues > 20:
            click.echo(f"    ⚠ Large group ({num_issues} issues) - this may be expensive", err=True)

        result = analyze_duplicate_group(group["issues"], client)

        if result["is_duplicate"] and result["confidence"] >= min_confidence:
            # Add AI metadata to the group
            group["ai_analysis"] = result
            filtered_groups.append(group)
            click.echo(f"    ✓ Duplicate (confidence: {result['confidence']:.2f})", err=True)
        else:
            click.echo(
                f"    ✗ Not duplicate (confidence: {result['confidence']:.2f}): {result['reasoning']}",
                err=True,
            )

    click.echo(f"AI filtering: {len(groups)} → {len(filtered_groups)} groups", err=True)
    return filtered_groups
