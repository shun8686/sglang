#!/usr/bin/env python3
"""
Data integrity check for defect_db.json. Run by 'maintain' Step A.

Checks format, uniqueness, completeness, deterministic rule violations, and NPU relevance:
- Required fields, bug_id format, duplicate detection
- Category/severity enum validity and value ranges
- needs_review flag consistency (confidence<0.7 → needs_review=True)
- NPU signal detection (zero NPU keywords → flag for manual review)
- Orphan commits and source consistency

Note: Category-severity semantic consistency is subjective — handled by Agent Audit, not here.

Usage:
    python .claude/skills/npu-defect-mine/scripts/consistency_check.py
"""

import re
import subprocess
from collections import Counter

from common import DB_DIR, REPO_ROOT, load_json, now_iso, save_json

VALID_CATEGORIES = {
    "crash",
    "precision_loss",
    "perf_regression",
    "compatibility",
    "compile_error",
}
VALID_SEVERITIES = {1, 3, 10}  # Minor, Major, Critical
REQUIRED_FIELDS = [
    "bug_id",
    "commit_sha",
    "category",
    "severity",
    "root_cause",
    "rationale",
    "confidence",
]
BUG_ID_RE = re.compile(r"^BUG-\d{4}-\d{3}$")
NPU_SIGNAL_RE = re.compile(
    r"\bnpu\b|\bascend\b|\bCANN\b|\bHCCL\b|torch_npu|sgl_kernel_npu", re.I
)


def check(db: dict) -> dict:
    defects = db.get("defects", [])
    issues = []

    # ── Per-defect checks ──
    seen_bug_ids = set()
    seen_pr_numbers = Counter()

    for d in defects:
        bug_id = d.get("bug_id", "MISSING")
        reasons = []

        # 1. Required fields present
        missing = [f for f in REQUIRED_FIELDS if f not in d or d[f] in (None, "")]
        if missing:
            reasons.append(f"missing fields: {missing}")

        # 2. bug_id format
        if not BUG_ID_RE.match(bug_id):
            reasons.append(f"invalid bug_id: {bug_id}")

        # 3. Duplicate bug_id
        if bug_id in seen_bug_ids:
            reasons.append(f"duplicate bug_id: {bug_id}")
        seen_bug_ids.add(bug_id)

        # 4. Category enum
        cat = d.get("category", "")
        if cat and cat not in VALID_CATEGORIES:
            reasons.append(f"invalid category: {cat}")

        # 5. Severity enum
        sev = d.get("severity", -1)
        if sev not in VALID_SEVERITIES:
            reasons.append(
                f"invalid severity: {sev} (expected one of {VALID_SEVERITIES})"
            )

        # 6. Confidence range
        conf = d.get("confidence", -1)
        if not (0 <= conf <= 1):
            reasons.append(f"confidence out of range: {conf}")

        # 6b. needs_review must be True when confidence < 0.7
        if conf < 0.7 and not d.get("needs_review", True):
            reasons.append(f"low confidence ({conf}) but needs_review=False")

        # 7. agent_version
        av = d.get("agent_version", "")
        if av in ("", "pending_agent_v1", "heuristic_v1", "heuristic_v2"):
            reasons.append(f"unclassified: agent_version={av or 'missing'}")

        # 8. source consistency
        if d.get("source") == "pr" and not d.get("pr_number"):
            reasons.append("source=pr but no pr_number")
        if d.get("pr_number"):
            seen_pr_numbers[d["pr_number"]] += 1

        # 9. NPU relevance signal (warning — requires manual review)
        npu_text = (
            (d.get("pr_title", "") or "")
            + " "
            + (d.get("pr_body", "") or "")[:5000]
            + " "
            + " ".join(d.get("files_fixed", []))
            + " "
            + " ".join(d.get("pr_labels", []))
        )
        if not NPU_SIGNAL_RE.search(npu_text):
            reasons.append(
                "no NPU signal in title/body/files/labels — possible non-NPU defect"
            )

        # 10. Orphan commit
        sha = d.get("commit_sha", "")
        if sha:
            r = subprocess.run(
                ["git", "merge-base", "--is-ancestor", sha, "origin/main"],
                capture_output=True,
                timeout=10,
                cwd=REPO_ROOT,
            )
            if r.returncode == 1:
                reasons.append("orphan commit (run git fetch && re-check)")
            elif r.returncode > 1:
                reasons.append(f"git error (rc={r.returncode}) for {sha[:10]}")
        else:
            reasons.append("missing commit_sha")

        if reasons:
            issues.append(f"{bug_id}: {'; '.join(reasons)}")

    # ── Cross-defect checks ──
    dup_prs = [pr for pr, count in seen_pr_numbers.items() if count > 1]
    if dup_prs:
        issues.append(f"CROSS: duplicate pr_number: {dup_prs}")

    return {
        "issues": issues,
        "issue_count": len(issues),
        "total_defects": len(defects),
    }


def main():
    db = load_json(DB_DIR / "defect_db.json")
    result = check(db)

    print(f"Integrity check: {result['total_defects']} defects")
    print(f"  Issues found: {result['issue_count']}")

    if result["issues"]:
        print(f"\nIssues:")
        for v in result["issues"]:
            print(f"  {v}")
    else:
        print("\nAll checks passed — clean")

    db["_meta"]["integrity_check_at"] = now_iso()
    db["_meta"]["integrity_issues"] = result["issue_count"]
    save_json(DB_DIR / "defect_db.json", db)


if __name__ == "__main__":
    main()
