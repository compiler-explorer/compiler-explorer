# Compiler Explorer Issue Analysis - Phase 1 Findings
**Date:** 2025-10-06
**Total Open Issues:** 855

## Executive Summary

The issue backlog reveals clear patterns that can guide systematic triage:

- **46% (398) are stale** (>2 years old, <5 comments) - prime candidates for review
- **35% (297) have zero engagement** - likely duplicates, invalid, or need clarification
- **48% (412) are poorly labeled** - only generic labels, impeding organization
- **51% (439) are generic "requests"** - mostly compiler/library additions
- **28% (239) are bugs** - actual functionality issues
- **5% (42) marked "probably-wont-happen"** but still open - should be closed kindly

## Key Statistics

### Age Distribution
```
>5 years old:    137 (16.0%)  ← Ancient, likely obsolete or philosophical
3-5 years old:   189 (22.1%)  ← Old, likely stale
2-3 years old:   181 (21.2%)  ← Getting stale
1-2 years old:   151 (17.7%)
6-12 months:     102 (11.9%)
3-6 months:       53 (6.2%)
<3 months:        42 (4.9%)   ← Recent
```

**Key Insight:** 59% of issues are >2 years old. This suggests either low prioritization or lack of contributor bandwidth.

### Label Distribution (Top 10)
```
request                 439 (51.3%)  ← Mostly compiler/library requests
bug                     239 (28.0%)
new-compilers           121 (14.2%)
new-libs                 93 (10.9%)
enhancement              79 (9.2%)
ui                       51 (6.0%)
help wanted              44 (5.1%)
probably-wont-happen     42 (4.9%)
Status: triaged          35 (4.1%)
lang-c++                 29 (3.4%)
```

### Engagement Patterns

**Top 3 Most-Commented:**
1. #82 (39 comments) - "C(++) Compiler Master List" - META-ISSUE tracking compilers
2. #891 (37 comments) - "Add pgroup compiler"
3. #264 (35 comments) - "Add D8 JavaScript assembly output"

**Zero Comments:** 297 issues (35%) have NO engagement whatsoever
- Many are recent (<6 months) and may just need time
- Many are 5+ years old and likely forgotten/obsolete

## Critical Findings

### 1. Tracking/Meta Issues

**Issue #82** (10 years old) is a "Master List" with checkboxes tracking compiler support. This was useful historically but now creates confusion:
- 39 comments of various requests
- Acts as a catch-all for "add this compiler" requests
- Should probably be closed with pointer to proper issue templates

### 2. "Probably Won't Happen" Issues (42 total)

These are already flagged but remain open. Examples:
- #187 (8.9y) - "Add support for MMIX" - niche educational architecture
- #264 (8.7y) - "Add D8 JavaScript assembly" - 35 comments, out of scope
- #341 (8.5y) - "classic GCC versions" - limited value vs maintenance cost
- #514 (8.1y) - "ARM GCC 6.3.0 standard libraries missing" - old compiler version

**Action:** These should be kindly closed with explanations.

### 3. Stale Issues (398 total)

**Criteria:** >2 years old AND <5 comments

These show minimal community interest and are likely:
- Overtaken by events (e.g., new compiler versions available)
- Niche requests with no PR momentum
- Questions that were never answered

**Sample oldest stale issues:**
- #187 (8.9y, 0 comments) - MMIX support
- #297 (8.6y, 0 comments) - GCC7 verbose-asm format
- #341 (8.5y, 0 comments) - Classic GCC versions
- #425 (8.4y, 0 comments) - libfirm/cparser

### 4. Duplicate/Similar Requests

**GCC versions:** 11 issues requesting various GCC versions/configurations
**Clang variants:** 9 issues for different Clang forks/versions
**MSVC versions:** 3+ issues for recent MSVC versions (could be consolidated)
**ROCm/AMD:** 3 issues about ROCm compiler versions

These could be consolidated into tracking issues or closed as duplicates.

### 5. Poorly Labeled Issues (412 total)

Many issues have only generic labels like "request" or "bug" without specifics:
- No language tag (lang-c++, lang-rust, etc.)
- No area tag (ui, compiler-issue, etc.)
- No priority indication

This makes filtering and prioritization difficult.

## Common Themes in Titles

**Top keywords:**
- "request" (437) - the label shows up in titles too
- "compiler" (183)
- "support" (97)
- "clang" (62)
- "library" (35)
- "msvc" (27)

**Pattern:** Most issues are straightforward "add X compiler" or "add Y library" requests.

## Recommendations for Phase 2

### Immediate Actions

1. **Close "Probably Won't Happen" Issues (42)**
   - Draft kind, clear explanation template
   - Close with rationale and link to contribution docs if applicable
   - Expected impact: -5% issue count

2. **Close Ancient Zero-Comment Issues (estimate ~50-100)**
   - Issues >5 years old with 0 comments and no activity
   - Use "stale bot" type reasoning: "no activity, assumed no longer relevant"
   - Allow 2 weeks for objections before closing
   - Expected impact: -6-12% issue count

3. **Close Obvious Duplicates (estimate ~20-30)**
   - Multiple GCC/Clang version requests → consolidate or close with "use latest"
   - Expected impact: -2-4% issue count

### Medium-Term Actions

4. **Improve Labeling (412 issues)**
   - Add language-specific labels (lang-*)
   - Add area labels (ui, execution, api, etc.)
   - Add priority labels where clear
   - This enables better filtering and assignment

5. **Create "Good First Issue" Pipeline (~50 issues)**
   - Review "help wanted" (44 issues) for suitable ones
   - Look for well-defined, isolated tasks
   - Add good-first-issue label and mentorship notes

6. **Consolidate Request Tracking**
   - Create wiki/doc page for "Compiler Request Guidelines"
   - Explain: how to request, how to contribute, what makes requests likely
   - Close #82 meta-issue with pointer to new docs

### Long-Term Strategy

7. **Implement Stale Bot**
   - Auto-label issues >1 year old with no activity
   - Auto-close after warning period if still no activity
   - Keep it friendly: "seems resolved, please reopen if still relevant"

8. **Improve Issue Templates**
   - Separate templates for: compiler requests, library requests, bug reports, feature requests
   - Require more details (version numbers, error messages, why needed)
   - Auto-label based on template used

9. **Regular Triage Cadence**
   - Weekly review of new issues (label, prioritize, close if needed)
   - Monthly review of old issues (close stale, consolidate duplicates)
   - Quarterly review of "help wanted" (ensure still relevant)

## Estimated Impact

**Conservative cleanup estimate:**
- Close "probably-wont-happen": -42 issues
- Close ancient zero-comment: -50 issues
- Close duplicates: -20 issues
- **Total: -112 issues (13% reduction)**

**Aggressive cleanup estimate:**
- Above plus stale >3 years: -200+ additional
- **Total: -300+ issues (35% reduction)**

## Next Steps

**Proposed Plan:**
1. Review sample of "probably-wont-happen" issues together
2. Draft closure message templates (kind but clear)
3. Review sample of stale issues together
4. Implement Phase 2: Systematic triage with your approval
5. Consider automation (stale bot, better templates)

## Files Generated

- `all-issues-raw.json` - Full issue data from GitHub
- `issues-summary.json` - Simplified format
- `full-analysis.txt` - Complete statistical analysis
- `PHASE1-FINDINGS.md` - This document
