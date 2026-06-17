#!/usr/bin/env python3
"""
PR-first defect extraction. Replaces extract_historical.py.

Strategy:
  1. Search merged PRs with NPU/Ascend labels → fetch full data
  2. Search merged PRs with fix/bug + NPU keywords in title → fetch full data
  3. git log origin/main as fallback for commits without PRs → gh search prs per SHA
  4. Output structured PR-rich data ready for Agent classification

Usage:
    python .claude/skills/npu-defect-mine/scripts/extract_prs.py [--since 2025-01-01] [--output pr_defects.json]
"""

import json
import re
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

from common import DB_DIR, REPO_ROOT, get_current_commit, load_json, now_iso, save_json

REPO = "sgl-project/sglang"
SINCE = "2025-01-01"

# ============================================================
# Helpers
# ============================================================


def gh_api(endpoint: str, jq_filter: str = ".") -> dict | None:
    """Call gh api with jq filter, return parsed JSON or None."""
    try:
        result = subprocess.run(
            ["gh", "api", endpoint, "--jq", jq_filter],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def gh_search_prs(query: str, limit: int = 200) -> list[dict]:
    """Search merged PRs, return list of {number, title, labels}."""
    items = gh_api(
        f"search/issues?q=repo:{REPO}+is:pr+is:merged+{query}&per_page={min(limit,100)}&sort=updated&order=desc",
        jq_filter="[.items[] | {number, title, labels: [.labels[].name], url}]",
    )
    return items or []


def gh_pr_detail(pr_number: int) -> dict | None:
    """Fetch full PR data: body, merge_commit_sha, merged_at, files, comments.
    Includes created_at and head.sha for unmerged PR fallback."""
    detail = gh_api(
        f"repos/{REPO}/pulls/{pr_number}",
        jq_filter="{number, title, body, state, merged_at, created_at, "
        "merge_commit_sha, head: {sha: .head.sha}, "
        "base: .base.ref, labels: [.labels[].name], url}",
    )
    return detail


def gh_pr_files(pr_number: int) -> list[str]:
    """Get list of files changed in a PR."""
    files = gh_api(
        f"repos/{REPO}/pulls/{pr_number}/files?per_page=50", jq_filter="[.[].filename]"
    )
    return files or []


def gh_pr_reviews(pr_number: int) -> list[dict]:
    """Get review comments for a PR."""
    reviews = gh_api(
        f"repos/{REPO}/pulls/{pr_number}/comments?per_page=30",
        jq_filter="[.[] | {user: .user.login, body, path, created_at}]",
    )
    return reviews or []


def gh_search_pr_for_sha(sha: str) -> dict | None:
    """Search for a PR containing a specific commit SHA."""
    result = gh_api(
        f"search/issues?q=repo:{REPO}+type:pr+sha:{sha[:10]}&per_page=1",
        jq_filter=".items[0] | {number, title, labels: [.labels[].name]}",
    )
    return result


# ============================================================
# NPU relevance validation
# ============================================================

# Word-boundary regex for substantive NPU/Ascend mentions (not substring matches)
_NPU_SIGNAL_RE = re.compile(
    r"\bnpu\b|\bascend\b|\bcann\b|\bhccl\b|torch_npu|sgl_kernel_npu"
    r"|\[NPU\]|\[Ascend\]|\[npu\]|\[ascend\]",
    re.I,
)
# Patterns that look like NPU signals but aren't:
#  - "input" contains "npu" → word boundary prevents this
#  - "cannot" contains "cann" → word boundary prevents this
#  - Launch command boilerplate: --device npu --attention-backend ascend


def is_genuinely_npu_defect(
    title: str, labels: list[str], files: list[str], body: str = ""
) -> tuple[bool, str]:
    """Check if a defect is genuinely NPU/Ascend-related.

    Returns (is_npu, reason).
    Must have at least ONE of:
      1. 'npu' or 'ascend' label
      2. [NPU] or [Ascend] tag in title
      3. 'npu' or 'ascend' in file path (e.g. hardware_backend/npu/)
      4. Substantive NPU discussion in PR body (not just launch command examples)
    """
    # Check 1: Labels
    labels_lower = [l.lower() for l in labels]
    if any("npu" in l or "ascend" in l for l in labels_lower):
        return True, "has NPU label"

    # Check 2: Title tag
    if re.search(r"\[NPU\]|\[Ascend\]|\[npu\]|\[ascend\]", title):
        return True, "has [NPU]/[Ascend] title tag"

    # Check 3: File paths
    if any("npu" in f.lower() or "ascend" in f.lower() for f in files):
        return True, "has NPU file path"

    # Check 4: Body — use word-boundary regex, exclude boilerplate sections
    if body:
        # Strip common boilerplate sections that contain launch commands
        clean_body = re.sub(
            r"## (?:Accuracy Tests|Speed Tests and Profiling).*?(?=## |\Z)",
            "",
            body,
            flags=re.DOTALL | re.I,
        )
        if _NPU_SIGNAL_RE.search(clean_body):
            return True, "substantive NPU discussion in body"

    return False, "no NPU signal (title, labels, files, body)"


# ============================================================
# Extraction
# ============================================================


def extract_single_pr(pr_number: int) -> list[dict]:
    """Extract a single PR by number, regardless of merge status.

    Skips state/merged/base checks — useful for analyzing PRs before merge
    (e.g. PR #23685 which is still OPEN).
    Also skips NPU signal check — the user explicitly chose this PR.

    Returns a list with one defect entry, or empty list if the PR fails
    content checks (doc-only, non-product, revert).
    """
    import re as _re

    print(f"=== Single PR Extraction: #{pr_number} ===")
    print(f"Repo: {REPO}\n")

    # Fetch PR detail — no state/merged/base filter
    detail = gh_pr_detail(pr_number)
    if not detail:
        print(f"ERROR: Could not fetch PR #{pr_number}")
        return []

    title = detail.get("title", "")
    state = detail.get("state", "unknown")
    merged = detail.get("merged_at")
    base = detail.get("base", "unknown")
    print(f"  PR #{pr_number}: {title[:80]}...")
    print(f"  State: {state} | Merged: {bool(merged)} | Base: {base}")

    # Skip reverts (they undo a fix, not introduce one)
    if title.lower().startswith("revert"):
        print(f"  SKIP: Revert PR")
        return []

    # Fetch files
    files = gh_pr_files(pr_number)

    # Skip pure documentation
    if files and all(
        f.startswith("docs/") or f.endswith((".md", ".rst", ".txt")) for f in files
    ):
        print(f"  SKIP: Documentation-only changes")
        return []

    # Skip non-product changes
    PRODUCT_PATHS = ("python/", "sgl-kernel/", "sgl-router/")
    if files and not any(f.startswith(PRODUCT_PATHS) for f in files):
        print(f"  SKIP: Non-product code (no python/sgl-kernel/sgl-router files)")
        return []

    # NPU relevance — skip in single-PR mode (user explicitly chose this PR).
    # Still log the signal analysis for informational purposes.
    body_text = (detail.get("body") or "")[:5000]
    labels = detail.get("labels", [])
    _, npu_reason = is_genuinely_npu_defect(title, labels, files, body_text)
    print(
        f"  NPU signal: {npu_reason} (informational only — not filtering in --pr mode)"
    )

    # Fetch reviews
    reviews = gh_pr_reviews(pr_number)

    # Build defect record — use sequential numbering (same format as append_daily)
    # to avoid format mismatch when both scripts target the same DB.
    date_str = (detail.get("merged_at") or detail.get("created_at") or now_iso())[:10]
    year = date_str[:4]
    next_num = 1
    existing_path = DB_DIR / "pr_defects.json"
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            for d in existing.get("defects", []):
                m = _re.match(r"BUG-\d{4}-(\d{3})", d.get("bug_id", ""))
                if m:
                    next_num = max(next_num, int(m.group(1)) + 1)
        except Exception:
            pass
    bug_id = f"BUG-{year}-{next_num:03d}"

    defect = {
        "bug_id": bug_id,
        "source": "pr",
        "pr_number": pr_number,
        "pr_title": title,
        "pr_body": body_text,
        "pr_labels": labels,
        "pr_url": detail.get("url", ""),
        "review_count": len(reviews),
        "reviews": reviews[:10],
        "commit_sha": detail.get("merge_commit_sha", "")
        or detail.get("head", {}).get("sha", ""),
        "date_fixed": detail.get("merged_at") or detail.get("created_at", ""),
        "files_fixed": files,
        # Mark as unmerged so downstream can treat differently
        "_unmerged": not bool(merged),
        # Classification fields (filled by Agent later)
        "category": "unknown",
        "severity": 0,
        "audit_count": 0,
        "root_cause": "unknown",
        "confidence": 0.0,
        "agent_version": "pending_agent_v1",
        "needs_review": True,
    }

    print(f"  [OK] Extracted: {bug_id}")
    print(f"  NPU reason: {npu_reason}")
    print(f"  Labels: {labels}")
    print(f"  Files: {len(files)} files")
    return [defect]


def extract_pr_defects(since: str = SINCE, max_prs: int = 300) -> list[dict]:
    """PR-first extraction: collect NPU defects from merged PRs."""
    print("=== PR-First Defect Extraction ===")
    print(f"Repo: {REPO}, Since: {since}\n")

    # ---- Step 1: Collect PRs via label search ----
    print("[1/4] Searching merged PRs by NPU labels...")
    seen_prs = OrderedDict()  # pr_number -> basic info, preserves order

    # Label-based searches
    label_queries = [
        ("label:npu", "npu-labeled"),
        ("label:ascend", "ascend-labeled"),
    ]
    for query, label in label_queries:
        prs = gh_search_prs(query, limit=250)
        for pr in prs:
            num = pr.get("number")
            if num and num not in seen_prs:
                seen_prs[num] = pr
        print(f"  {label}: {len(seen_prs)} unique so far")
        time.sleep(1)

    # ---- Step 2: Keyword title search for PRs without NPU labels ----
    print("[2/4] Searching merged PRs by NPU keywords in title...")
    keyword_queries = [
        '"NPU"+"fix"',
        '"NPU"+"bug"',
        '"Ascend"+"fix"',
        '"CANN"+"fix"',
    ]
    for query in keyword_queries:
        prs = gh_search_prs(query, limit=100)
        added = 0
        for pr in prs:
            num = pr.get("number")
            if num and num not in seen_prs:
                seen_prs[num] = pr
                added += 1
        print(f"  {query}: +{added} ({len(seen_prs)} total)")
        time.sleep(2)  # slower rate limit for search API

    before_fallback = len(seen_prs)
    print(f"\n  Total PR candidates: {len(seen_prs)}")

    # ---- Step 3: Fallback — git log for commits without PRs ----
    print("[3/4] Fallback: git log for commits not covered by PRs...")
    cmd = [
        "git",
        "log",
        "--no-merges",
        "origin/main",
        f"--since={since}",
        "--format=%H|%ai|%s",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
        timeout=120,
    )

    # NPU keywords for filtering — word boundaries prevent substring false positives:
    #  - "npu" matches inside "input" (i-n-p-u-t)  →  \bnpu\b fixes this
    #  - "cann" matches inside "cannot", "scanning"  →  \bCANN\b fixes this
    npu_kw = re.compile(
        r"\bnpu\b|\bascend\b|\bCANN\b|\bHCCL\b|torch_npu|sgl_kernel_npu", re.I
    )
    fix_kw = re.compile(
        r"\bfix\b|\bbug\b|\brepair\b|\bcorrect\b|\bpatch\b|\bresolve\b|\bprevent\b",
        re.I,
    )

    commits = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        sha, date_str, msg = parts
        if fix_kw.search(msg) and npu_kw.search(msg):
            commits.append(
                {"sha": sha.strip(), "date": date_str.strip(), "message": msg.strip()}
            )

    print(f"  NPU fix commits on origin/main: {len(commits)}")

    # Try to link each commit to a PR
    fallback_found = 0
    for i, c in enumerate(commits):
        sha = c["sha"]
        # Skip if already covered by a known PR
        pr_match = re.search(r"#(\d{4,6})", c["message"])
        if pr_match:
            pr_num = int(pr_match.group(1))
            if pr_num in seen_prs:
                continue

        # gh search prs for this commit
        if i > 0 and i % 10 == 0:
            print(f"    ... {i}/{len(commits)} ({fallback_found} linked to PRs)")
            time.sleep(1.5)

        pr = gh_search_pr_for_sha(sha)
        if pr and pr.get("number"):
            num = pr["number"]
            if num not in seen_prs:
                seen_prs[num] = pr
                fallback_found += 1

    print(f"  Fallback linked: {fallback_found} new PRs")

    # ---- Step 4: Fetch full data for each PR ----
    print(f"\n[4/4] Fetching full data for {len(seen_prs)} unique PRs...")
    defects = []
    skipped = 0
    fetch_errors = 0

    pr_list = list(seen_prs.values())
    if max_prs > 0 and len(pr_list) > max_prs:
        print(f"  Limiting to {max_prs} PRs (--max-prs)")
        pr_list = pr_list[:max_prs]

    for i, pr_info in enumerate(pr_list):
        pr_num = pr_info.get("number")
        if not pr_num:
            skipped += 1
            continue

        if i > 0 and i % 20 == 0:
            print(f"  ... {i}/{len(pr_list)} ({len(defects)} extracted)")
            time.sleep(1.5)  # rate limit
        elif i > 0 and i % 5 == 0:
            time.sleep(0.5)

        # Fetch PR detail
        detail = gh_pr_detail(pr_num)
        if not detail:
            fetch_errors += 1
            continue

        # Skip non-merged or non-main-base PRs
        if detail.get("state") != "closed" or not detail.get("merged_at"):
            skipped += 1
            continue
        if detail.get("base") != "main":
            skipped += 1
            continue

        # Skip reverts (they undo a fix, not introduce one)
        title = detail.get("title", "")
        if title.lower().startswith("revert"):
            skipped += 1
            continue

        # Fetch files early (needed for doc-only, non-product, and NPU checks below)
        files = gh_pr_files(pr_num)

        # Skip pure documentation (no source code)
        if files and all(
            f.startswith("docs/") or f.endswith((".md", ".rst", ".txt")) for f in files
        ):
            skipped += 1
            continue

        # Skip non-product changes (no source code)
        PRODUCT_PATHS = ("python/", "sgl-kernel/", "sgl-router/")
        if files and not any(f.startswith(PRODUCT_PATHS) for f in files):
            skipped += 1
            continue

        # Skip if title doesn't suggest a defect fix
        title = detail.get("title", "")
        if not fix_kw.search(title):
            # Check labels as alternative signal
            labels = detail.get("labels", [])
            if not any(
                kw in str(labels).lower() for kw in ("bug", "fix", "bugfix", "critical")
            ):
                skipped += 1
                continue

        # Fetch reviews (files already fetched above)
        reviews = gh_pr_reviews(pr_num)

        # Build defect record — validate NPU relevance first
        body_text = (detail.get("body") or "")[:5000]
        labels = detail.get("labels", [])
        is_npu, npu_reason = is_genuinely_npu_defect(title, labels, files, body_text)
        if not is_npu:
            print(f"  SKIP PR#{pr_num}: not NPU ({npu_reason})")
            skipped += 1
            continue

        bug_id = f"BUG-{detail['merged_at'][:4]}-{len(defects)+1:03d}"

        defects.append(
            {
                "bug_id": bug_id,
                "source": "pr",
                "pr_number": pr_num,
                "pr_title": title,
                "pr_body": body_text,
                "pr_labels": labels,
                "pr_url": detail.get("url", ""),
                "review_count": len(reviews),
                "reviews": reviews[:10],  # store top 10 reviews
                "commit_sha": detail.get("merge_commit_sha", ""),
                "date_fixed": detail.get("merged_at", ""),
                "files_fixed": files,
                # Classification fields (filled by Agent later)
                "category": "unknown",
                "severity": 0,
                "audit_count": 0,
                "root_cause": "unknown",
                "confidence": 0.0,
                "agent_version": "pending_agent_v1",
                "needs_review": True,
            }
        )

    print(f"\n=== Extraction Complete ===")
    print(f"  PRs found:    {len(seen_prs)}")
    print(f"  Extracted:    {len(defects)}")
    print(f"  Skipped:      {skipped} (not merged to main / not a fix)")
    print(f"  Fetch errors: {fetch_errors}")
    print(f"  Fallback PRs: {fallback_found}")

    return defects


# ============================================================
# Main
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PR-first NPU defect extraction")
    parser.add_argument("--since", default=SINCE, help="Start date for search")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--max-prs", type=int, default=300, help="Max PRs to process")
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="Extract a single PR by number (skips merge-status filter)",
    )
    args = parser.parse_args()

    if args.pr:
        # Single PR mode: extract one PR regardless of merge status
        defects = extract_single_pr(args.pr)
        if not defects:
            print(
                f"\nNo defect extracted for PR #{args.pr} "
                f"(check: NPU relevance, doc-only, non-product, or fetch error)"
            )
            return
    else:
        defects = extract_pr_defects(since=args.since, max_prs=args.max_prs)

    output_path = Path(args.output) if args.output else (DB_DIR / "pr_defects.json")
    save_json(
        output_path,
        {
            "extracted_at": now_iso(),
            "since": args.since,
            "git_commit": get_current_commit(),
            "total_defects": len(defects),
            "defects": defects,
        },
    )

    print(f"\nSaved {len(defects)} defects to: {output_path}")
    print(
        "\nNext: python .claude/skills/npu-defect-mine/scripts/seed_defect_db.py --agent --from-prs"
    )


if __name__ == "__main__":
    main()
