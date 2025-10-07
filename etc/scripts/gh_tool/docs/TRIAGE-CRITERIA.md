# Compiler Explorer Issue Triage Criteria

Based on manual review of 22+ issues (October 2025), here are the established triage patterns and decisions.

## Quick Reference: Triage Decisions

| Category | Keep/Close | Labels to Add |
|----------|-----------|---------------|
| Active projects (compiler/lib requests) | KEEP | help wanted |
| Abandoned upstream projects (8+ years) | CLOSE | - |
| Zero-demand features (7+ years, 0 comments) | CLOSE | - |
| External tools (already work via API) | CLOSE | - |
| Stale reminders/TODOs (6+ years) | CLOSE | - |
| Duplicates | CLOSE newest | - |
| Related but not duplicates | KEEP both | Link in comments |

---

## 1. Core Principles

### Conservative Approach
- **Age alone is NOT a proxy for value**
- Each issue needs explicit review before closing
- When in doubt, keep it open
- Look for recent interest (CppCon mentions, user questions, etc.)

### Priority Areas
1. Close obvious duplicates
2. Close abandoned/dead projects
3. Better organize remaining issues with labels
4. Identify "help wanted" opportunities

---

## 2. Compiler & Library Requests

### KEEP if:
- ✅ Upstream project is actively maintained (check last commit date)
- ✅ Project has legitimate use cases
- ✅ Zero comments but project is real/active (just needs contributor)

### CLOSE if:
- ❌ Upstream project is abandoned (8+ years with no activity)
- ❌ Author confirmed they don't have patches/support for newer versions
- ❌ Project was experimental and never went mainstream

### Examples:

**KEEP:**
- #425 (libfirm/cparser) - Active: cparser updated Jan 2025, libfirm March 2025
- #1232 (PCL Point Cloud Library) - Active, legitimate library
- #503 (Cython) - Active project, add "help wanted"
- #341 (classic GCC versions) - User interest confirmed at CppCon

**CLOSE:**
- #2063 (static_print GCC) - Abandoned (last commit 2017), author has no patches
- #1018 (clang-contracts) - CE already has alternative contract compilers

---

## 3. Feature Requests

### KEEP if:
- ✅ Technically feasible
- ✅ Aligns with CE's mission
- ✅ Recent interest or discussion
- ✅ "Probably-wont-happen" but PRs would be accepted

### CLOSE if:
- ❌ Implementation complexity outweighs value
- ❌ Encourages bad practices (e.g., #797 BUILT_ON_COMPILER_EXPLORER macro)
- ❌ Tool already exists externally and works via CE's API

### Examples:

**KEEP:**
- #297 (-fverbose-asm filtering bug) - Real usability issue, fixable
- #1004 (verbose-asm UI button) - Related to #297, useful convenience
- #187 (MMIX) - Recent CppCon interest

**CLOSE:**
- #547 (FXC shader compiler) - Not a good fit for CE
- #797 (BUILT_ON_COMPILER_EXPLORER macro) - Encourages bad practices
- #1203 (SIMD-Visualiser integration) - Already works externally via API

---

## 4. Bugs & Issues

### Bug Validation Process:
1. Test if bug still reproduces using Compiler Explorer MCP
2. Read ALL comments, not just the first one
3. Check if workarounds exist
4. Verify affected compiler versions still exist in CE

### KEEP if:
- ✅ Bug still reproduces
- ✅ Affects multiple compiler versions
- ✅ No reasonable workaround exists

### Examples:

**KEEP:**
- #514 (ARM GCC hard float) - Tested, still fails with `-mfloat-abi=hard` on 6 ARM GCC versions

**CLOSE:**
- Issues that were already fixed but not closed
- Issues with trivial workarounds

---

## 5. Internal Tasks & Reminders

### CLOSE if:
- ❌ 6+ year old reminder with no action taken
- ❌ Decision was implicitly made (kept the code/feature)
- ❌ No longer relevant to current codebase

### Examples:

**CLOSE:**
- #1357 (binutils-multi reminder) - 6.5 years old, still installed, serves a purpose
- Accidental duplicates from batch operations

---

## 6. Related vs Duplicate Issues

### Not Duplicates if:
- Issues address different aspects of the same problem
- One is prerequisite for the other
- Different implementation approaches

### Duplicates if:
- Identical title and description
- Same request posted multiple times
- Accidental double-posts

### Examples:

**Related (NOT duplicates):**
- #297 (fix filter bug) + #1004 (add UI button) - Related but distinct

**Duplicates (CLOSE):**
- #3201, #7778, #8111 - All requesting NumPy
- #4336, #6526 - Both requesting Groovy
- #3426, #5972 - Both requesting OpenBLAS
- #7601, #7602 - Accidental double-post

---

## 7. Labeling Strategy

### Add "help wanted" when:
- Valid request but maintainer doesn't have time
- Clear scope, would accept PRs
- Not marked "probably-wont-happen"

### Keep "probably-wont-happen" when:
- Low priority but PRs would still be accepted
- Niche use case
- High complexity/low demand

### Remove "probably-wont-happen" when:
- Recent interest or demand emerges
- Project confirmed active and relevant
- Better candidate for "help wanted"

---

## 8. Label Changes Made

### Reviewed Issues - Actions Taken:

| Issue | Action | Labels | Reason |
|-------|--------|--------|--------|
| #264 | KEEP | Remove probably-wont-happen | CE supports interpreted languages |
| #187 | KEEP | Add help wanted | Recent CppCon interest |
| #503 | KEEP | Add help wanted | Active project, PRs welcome |
| #341 | KEEP | Add help wanted | User interest confirmed |
| #514 | KEEP | Remove probably-wont-happen | Bug still exists, affects 6 versions |
| #547 | CLOSE | - | Not a good fit |
| #797 | CLOSE | - | Encourages bad practices |
| #297 | KEEP | Add bug/usability | Filter breaks verbose-asm |
| #1004 | KEEP | Link to #297 | UI convenience, related to #297 |
| #1018 | CLOSE | - | Satisfied by alternative compilers |
| #2063 | CLOSE | - | Abandoned upstream |
| #425 | KEEP | Add help wanted | Active project |
| #668 | CLOSE | - | Zero demand, very niche |
| #1203 | CLOSE | - | Works externally already |
| #1232 | KEEP | Add help wanted | Legitimate library |
| #1357 | CLOSE | - | Stale reminder, implicitly resolved |
| #7778 | CLOSE | Duplicate of #3201 | NumPy duplicate |
| #8111 | CLOSE | Duplicate of #3201 | NumPy duplicate |
| #5972 | CLOSE | Duplicate of #3426 | OpenBLAS duplicate |
| #6526 | CLOSE | Duplicate of #4336 | Groovy duplicate |
| #7601 | CLOSE | Duplicate of #7602 | Accidental duplicate |

---

## 9. Triage Process Recommendations

### Monthly Triage Workflow:
1. **Identify duplicates** (automated similarity matching)
2. **Check upstream status** for compiler/library requests (last commit date)
3. **Test bugs** using Compiler Explorer MCP
4. **Review engagement** (comments, reactions, recent activity)
5. **Apply labels** (help wanted, good first issue, etc.)
6. **Close obvious cases** (duplicates, abandoned projects, stale reminders)

### Tools Used:
- GitHub API for bulk data extraction
- Python scripts for similarity analysis
- Compiler Explorer MCP for live testing
- Web search for upstream project status

---

## 10. Communication Templates

### Closing Duplicates:
```
Closing as duplicate of #XXXX which has more discussion/history.
Please follow that issue for updates.
```

### Closing Abandoned Projects:
```
Closing as the upstream project appears abandoned (last commit [date]).
If the project becomes active again, please feel free to reopen or create a new issue.
```

### Closing External Tools:
```
Closing as this tool already works externally using Compiler Explorer's public API.
Integration would not provide significant additional value.
See: [tool URL]
```

### Keeping with "help wanted":
```
This is a valid request but requires a contributor with expertise in [area].
Marking as "help wanted" - PRs are welcome!
```

---

## Statistics from This Triage Session

- **Issues reviewed**: 22
- **Issues to close**: 12 (5 duplicates + 7 other)
- **Issues to keep**: 10
- **"help wanted" added**: 5
- **Relationships identified**: 1 pair (#297 + #1004)

**Impact**:
- Closed duplicates: -5 issues
- Closed abandoned/unviable: -7 issues
- Net reduction: -12 issues (~1.4% of 855 total)
- Better organized: 10 issues now have clearer status

---

## Next Steps

1. Apply label changes to reviewed issues
2. Close duplicates with appropriate comments
3. Close abandoned/stale issues with appropriate comments
4. Link related issues (#297 ↔ #1004)
5. Continue triage with next batch of issues
6. Update this document as new patterns emerge
