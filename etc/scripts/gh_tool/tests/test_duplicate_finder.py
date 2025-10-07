# Copyright (c) 2025, Compiler Explorer Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gh_tool.duplicate_finder import calculate_similarity, find_duplicates, generate_report


def make_issue(number, title, created_at=None, comments=None):
    """Helper to create test issue data."""
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    if comments is None:
        comments = []

    return {
        "number": number,
        "title": title,
        "createdAt": created_at,
        "updatedAt": created_at,
        "state": "open",
        "labels": [],
        "comments": comments,
    }


class TestCalculateSimilarity:
    @pytest.mark.parametrize(
        "title1,title2,expected_min,expected_max",
        [
            # Identical strings
            ("Add GCC 13", "Add GCC 13", 1.0, 1.0),
            # Case insensitive
            ("Add GCC 13", "add gcc 13", 1.0, 1.0),
            # Completely different
            ("Add GCC 13", "Python bug", 0.0, 0.3),
            # Similar titles with different case
            ("[LIB REQUEST] numpy", "[LIB REQUEST] NumPy", 0.9, 1.0),
            # Partial similarity
            ("Add GCC 13 support", "Add GCC 14 support", 0.7, 1.0),
            # Different request types but same content (tags stripped)
            ("[COMPILER REQUEST] foo", "[LIB REQUEST] foo", 1.0, 1.0),
        ],
    )
    def test_similarity_calculation(self, title1, title2, expected_min, expected_max):
        similarity = calculate_similarity(title1, title2)
        assert expected_min <= similarity <= expected_max


class TestFindDuplicates:
    def test_no_duplicates(self):
        issues = [
            make_issue(1, "Add GCC 13"),
            make_issue(2, "Fix Python bug"),
            make_issue(3, "Update documentation"),
        ]
        groups = find_duplicates(issues, threshold=0.6)
        assert len(groups) == 0

    def test_simple_duplicate_pair(self):
        issues = [
            make_issue(1, "Add NumPy library"),
            make_issue(2, "Add numpy library"),
        ]
        groups = find_duplicates(issues, threshold=0.8)
        assert len(groups) == 1
        assert len(groups[0]["issues"]) == 2
        assert groups[0]["issues"][0]["number"] in [1, 2]
        assert groups[0]["issues"][1]["number"] in [1, 2]

    def test_multiple_duplicate_groups(self):
        issues = [
            make_issue(1, "Add GCC 13"),
            make_issue(2, "Add gcc 13"),
            make_issue(3, "Add Clang 17"),
            make_issue(4, "Add clang 17"),
            make_issue(5, "Fix bug"),
        ]
        groups = find_duplicates(issues, threshold=0.85)
        assert len(groups) == 2

    def test_transitive_grouping(self):
        # If A~B and B~C, they should all be in one group
        issues = [
            make_issue(1, "Add NumPy"),
            make_issue(2, "Add numpy"),
            make_issue(3, "Add NumPy library"),
        ]
        groups = find_duplicates(issues, threshold=0.7)
        # Should be grouped together transitively
        assert len(groups) >= 1
        # At least one group should have all 3 issues
        max_group_size = max(len(g["issues"]) for g in groups)
        assert max_group_size >= 2

    def test_age_filtering(self):
        now = datetime.now(UTC)
        old_date = (now - timedelta(days=100)).isoformat()
        recent_date = (now - timedelta(days=5)).isoformat()

        issues = [
            make_issue(1, "Add NumPy", created_at=old_date),
            make_issue(2, "Add numpy", created_at=recent_date),
        ]

        # Filter to only old issues (>30 days)
        groups = find_duplicates(issues, threshold=0.8, min_age_days=30)
        # Should only have one issue in the comparison, so no duplicates found
        assert len(groups) == 0

        # No age filter should find the duplicate
        groups = find_duplicates(issues, threshold=0.8, min_age_days=0)
        assert len(groups) == 1

    def test_threshold_sensitivity(self):
        issues = [
            make_issue(1, "Add GCC 13 compiler"),
            make_issue(2, "Add GCC 14 compiler"),
        ]

        # Low threshold should find similarity
        groups = find_duplicates(issues, threshold=0.6)
        assert len(groups) == 1

        # Very high threshold should not
        groups = find_duplicates(issues, threshold=0.99)
        assert len(groups) == 0


class TestGenerateReport:
    def test_empty_report(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            output_file = f.name

        try:
            generate_report([], output_file)
            content = Path(output_file).read_text()
            assert "No potential duplicates found" in content
        finally:
            Path(output_file).unlink()

    def test_report_with_groups(self):
        groups = [
            {
                "issues": [
                    make_issue(1, "Add NumPy", created_at="2020-01-01T00:00:00Z"),
                    make_issue(2, "Add numpy", created_at="2020-01-02T00:00:00Z"),
                ],
                "max_similarity": 0.95,
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            output_file = f.name

        try:
            generate_report(groups, output_file)
            content = Path(output_file).read_text()

            assert "# Potential Duplicate Issues" in content
            assert "Found 1 potential duplicate groups" in content
            assert "## Group 1 (95% similar)" in content
            assert "#1" in content
            assert "#2" in content
            assert "numpy" in content.lower()
        finally:
            Path(output_file).unlink()
