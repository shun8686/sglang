#!/usr/bin/env python3
"""
Daily Incremental Defect Appender (PR-first only).

Scans merged PRs with NPU labels since last_scan_at, fetches full PR data,
appends to defect_db.json, and optionally prepares Agent classification batch.

Usage:
    python .claude/skills/npu-defect-mine/scripts/append_daily.py [--agent] [--dry-run]
"""

import subprocess
import time

from common import (
    DB_DIR,
    get_current_commit,
    load_json,
    now_iso,
    save_json,
)

DEFECT_DB_PATH = DB_DIR / "defect_db.json"

EMPTY_DEFECT_DB = {
    "_meta": {
        "last_scan_commit": "",
        "last_scan_at": "",
        "total_defects": 0,
        "schema_version": "1.1",
    },
    "defects": [],
    "near_misses": [],
}


def ensure_defect_db():
    """Create defect_db.json if it doesn't exist."""
    if not DEFECT_DB_PATH.exists():
        DEFECT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = dict(EMPTY_DEFECT_DB)
        db["_meta"]["last_scan_commit"] = get_current_commit()
        db["_meta"]["last_scan_at"] = now_iso()
        save_json(DEFECT_DB_PATH, db)
        print("Initialized empty defect_db.json")
    return load_json(DEFECT_DB_PATH)


def scan_new_prs(db: dict) -> list[dict]:
    """Scan merged PRs with NPU labels since last scan.

    Uses gh search prs with date filter + label:npu / label:ascend,
    fetches full PR data (body, files, labels), validates NPU relevance,
    and returns defect records ready for append + Agent classification.
    """
    import json as _json
    import re as _re

    last_scan_at = db["_meta"].get("last_scan_at", "")
    if not last_scan_at:
        print("No last_scan_at — run init first.")
        return []

    since_date = last_scan_at[:10]  # YYYY-MM-DD
    print(f"Scanning PRs merged since {since_date}...")

    fix_kw = _re.compile(
        r"\bfix\b|\bbug\b|\brepair\b|\bpatch\b|\bresolve\b|\bprevent\b", _re.I
    )
    npu_signal_re = _re.compile(
        r"\bnpu\b|\bascend\b|\bcann\b|\bhccl\b|torch_npu|sgl_kernel_npu|\[NPU\]|\[Ascend\]",
        _re.I,
    )

    # Search merged PRs with NPU labels
    seen = set()
    all_prs = []

    for label in ("npu", "ascend"):
        result = subprocess.run(
            [
                "gh",
                "api",
                f"search/issues?q=repo:sgl-project/sglang+is:pr+is:merged+label:{label}+merged:>={since_date}&per_page=50&sort=updated&order=desc",
                "--jq",
                "[.items[] | {number, title, labels: [.labels[].name]}]",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            prs = _json.loads(result.stdout)
            for pr in prs:
                num = pr.get("number")
                if num and num not in seen and fix_kw.search(pr.get("title", "")):
                    seen.add(num)
                    all_prs.append(pr)
        time.sleep(1)

    print(f"  Found {len(all_prs)} merged fix PRs with NPU labels")

    # Determine next bug number from max existing (not total_defects after deletions)
    max_num = 0
    for d in db["defects"]:
        m = _re.match(r"BUG-\d{4}-(\d{3})", d.get("bug_id", ""))
        if m:
            max_num = max(max_num, int(m.group(1)))
    next_num = max_num + 1

    # Fetch full data for each PR
    defects = []
    for i, pr_info in enumerate(all_prs):
        pr_num = pr_info["number"]
        if i > 0 and i % 10 == 0:
            print(f"  ... {i}/{len(all_prs)}")
            time.sleep(1)

        detail = subprocess.run(
            [
                "gh",
                "api",
                f"repos/sgl-project/sglang/pulls/{pr_num}",
                "--jq",
                "{title, body, merged_at, merge_commit_sha, base: .base.ref, labels: [.labels[].name]}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        if detail.returncode != 0:
            continue
        d = _json.loads(detail.stdout)
        if d.get("base") != "main":
            continue
        if not d.get("merged_at"):
            continue

        title = d.get("title", "")
        if title.lower().startswith("revert"):
            continue
        date = d["merged_at"][:10]
        year = date[:4] if date else "UNKNOWN"

        # Get files
        files_result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/sgl-project/sglang/pulls/{pr_num}/files?per_page=30",
                "--jq",
                "[.[].filename]",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        files = (
            _json.loads(files_result.stdout)
            if files_result.returncode == 0 and files_result.stdout.strip()
            else []
        )

        # Skip pure documentation
        if files and all(
            f.startswith("docs/") or f.endswith((".md", ".rst", ".txt")) for f in files
        ):
            continue

        # Skip non-product changes
        PRODUCT_PATHS = ("python/", "sgl-kernel/", "sgl-router/")
        if files and not any(f.startswith(PRODUCT_PATHS) for f in files):
            continue

        # NPU relevance validation (defensive — label search guarantees NPU, verify anyway)
        labels = d.get("labels", [])
        has_npu_label = any(
            "npu" in lb.lower() or "ascend" in lb.lower() for lb in labels
        )
        has_npu_title = bool(
            _re.search(r"\[NPU\]|\[Ascend\]|\[npu\]|\[ascend\]", title, _re.I)
        )
        has_npu_file = any("npu" in f.lower() or "ascend" in f.lower() for f in files)
        # Check body — strip boilerplate Accuracy/Speed Tests sections before matching
        body_text = (d.get("body") or "")[:5000]
        clean_body = _re.sub(
            r"## (?:Accuracy Tests|Speed Tests and Profiling).*?(?=## |\Z)",
            "",
            body_text,
            flags=_re.DOTALL | _re.I,
        )
        has_npu_body = bool(npu_signal_re.search(clean_body))
        if not (has_npu_label or has_npu_title or has_npu_file or has_npu_body):
            print(f"  SKIP PR#{pr_num}: no NPU signal after validation")
            continue

        defects.append(
            {
                "bug_id": f"BUG-{year}-{next_num + len(defects):03d}",
                "source": "pr",
                "pr_number": pr_num,
                "pr_title": title,
                "pr_body": (d.get("body") or "")[:3000],
                "pr_labels": labels,
                "review_count": 0,
                "reviews": [],
                "commit_sha": d.get("merge_commit_sha", ""),
                "date_fixed": d.get("merged_at", ""),
                "files_fixed": files,
                "category": "unknown",
                "severity": 0,
                "audit_count": 0,
                "root_cause": "unknown",
                "confidence": 0.0,
                "agent_version": "pending_agent_v1",
                "needs_review": True,
            }
        )

    print(f"  -> {len(defects)} new defects extracted")
    return defects


def _scan_single_pr(pr_number: int, db: dict) -> list[dict]:
    """Scan a single PR by number, regardless of merge status and NPU signal.

    Fetches PR detail, and returns a defect record ready for append + Agent
    classification.  No state/merged/base/NPU-signal filtering — the user
    explicitly chose this PR for analysis.

    Useful for analyzing PRs before they are merged (e.g. PR #23685),
    or PRs that don't have NPU labels but are relevant to NPU risk analysis.
    """
    import json as _json
    import re as _re

    npu_signal_re = _re.compile(
        r"\bnpu\b|\bascend\b|\bcann\b|\bhccl\b|torch_npu|sgl_kernel_npu|\[NPU\]|\[Ascend\]",
        _re.I,
    )

    print(f"Fetching PR #{pr_number} (no merge-status filter)...")

    # Check if already in DB
    existing_prs = {d.get("pr_number") for d in db["defects"] if d.get("pr_number")}
    if pr_number in existing_prs:
        print(f"  PR #{pr_number} already exists in DB — skipping.")
        return []

    # Fetch PR detail via gh CLI
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/sgl-project/sglang/pulls/{pr_number}",
            "--jq",
            "{number, title, body, state, merged_at, created_at, "
            "merge_commit_sha, base: .base.ref, labels: [.labels[].name], "
            "head: {sha}, url}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )

    if result.returncode != 0 or not result.stdout.strip():
        print(f"  ERROR: Could not fetch PR #{pr_number}")
        return []

    detail = _json.loads(result.stdout)
    title = detail.get("title", "")
    state = detail.get("state", "unknown")
    merged = detail.get("merged_at")
    print(f"  Title: {title[:80]}...")
    print(
        f"  State: {state} | Merged: {bool(merged)} | Base: {detail.get('base', '?')}"
    )

    # Skip reverts
    if title.lower().startswith("revert"):
        print(f"  SKIP: Revert PR")
        return []

    # Get files
    files_result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/sgl-project/sglang/pulls/{pr_number}/files?per_page=30",
            "--jq",
            "[.[].filename]",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    files = (
        _json.loads(files_result.stdout)
        if files_result.returncode == 0 and files_result.stdout.strip()
        else []
    )

    # Skip pure documentation
    if files and all(
        f.startswith("docs/") or f.endswith((".md", ".rst", ".txt")) for f in files
    ):
        print(f"  SKIP: Documentation-only")
        return []

    # Skip non-product changes
    PRODUCT_PATHS = ("python/", "sgl-kernel/", "sgl-router/")
    if files and not any(f.startswith(PRODUCT_PATHS) for f in files):
        print(f"  SKIP: Non-product code")
        return []

    # NPU relevance validation
    labels = detail.get("labels", [])
    has_npu_label = any("npu" in lb.lower() or "ascend" in lb.lower() for lb in labels)
    has_npu_title = bool(
        _re.search(r"\[NPU\]|\[Ascend\]|\[npu\]|\[ascend\]", title, _re.I)
    )
    has_npu_file = any("npu" in f.lower() or "ascend" in f.lower() for f in files)
    body_text = (detail.get("body") or "")[:5000]
    clean_body = _re.sub(
        r"## (?:Accuracy Tests|Speed Tests and Profiling).*?(?=## |\Z)",
        "",
        body_text,
        flags=_re.DOTALL | _re.I,
    )
    has_npu_body = bool(npu_signal_re.search(clean_body))

    # In --pr mode, log NPU signals but don't filter — user explicitly chose this PR
    npu_signals = []
    if has_npu_label:
        npu_signals.append("label")
    if has_npu_title:
        npu_signals.append("title")
    if has_npu_file:
        npu_signals.append("file")
    if has_npu_body:
        npu_signals.append("body")
    print(
        f"  NPU signals: {npu_signals if npu_signals else 'none'} "
        f"(informational only — not filtering in --pr mode)"
    )

    # Determine bug ID
    max_num = 0
    for d in db["defects"]:
        m = _re.match(r"BUG-(\d{4})-(\d{3})", d.get("bug_id", ""))
        if m:
            max_num = max(max_num, int(m.group(2)))
    next_num = max_num + 1
    date_str = (detail.get("merged_at") or detail.get("created_at") or now_iso())[:10]
    year = date_str[:4]

    defect = {
        "bug_id": f"BUG-{year}-{next_num:03d}",
        "source": "pr",
        "pr_number": pr_number,
        "pr_title": title,
        "pr_body": (detail.get("body") or "")[:3000],
        "pr_labels": labels,
        "review_count": 0,
        "reviews": [],
        "commit_sha": detail.get("merge_commit_sha", "")
        or detail.get("head", {}).get("sha", ""),
        "date_fixed": detail.get("merged_at") or detail.get("created_at", ""),
        "files_fixed": files,
        "_unmerged": not bool(merged),
        "category": "unknown",
        "severity": 0,
        "audit_count": 0,
        "root_cause": "unknown",
        "confidence": 0.0,
        "agent_version": "pending_agent_v1",
        "needs_review": True,
    }

    print(f"  [OK] Extracted: {defect['bug_id']} (unmerged={defect['_unmerged']})")
    return [defect]


def append_to_db(db: dict, new_defects: list[dict]) -> dict:
    """Append new defects and update metadata. Deduplicates by pr_number/commit_sha."""
    existing_prs = {d.get("pr_number") for d in db["defects"] if d.get("pr_number")}
    existing_shas = {d.get("commit_sha") for d in db["defects"] if d.get("commit_sha")}
    deduped = []
    skipped = 0
    for d in new_defects:
        pr = d.get("pr_number")
        sha = d.get("commit_sha")
        if pr and pr in existing_prs:
            skipped += 1
            continue
        if sha and sha in existing_shas:
            skipped += 1
            continue
        deduped.append(d)
    if skipped:
        print(f"  Skipped {skipped} duplicate(s) already in DB")
    db["defects"].extend(deduped)
    db["_meta"]["last_scan_commit"] = get_current_commit()
    db["_meta"]["last_scan_at"] = now_iso()
    db["_meta"]["total_defects"] = len(db["defects"])
    return db


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Daily incremental defect appender (PR-first)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be added without saving"
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Auto-prepare Agent classification batch (pending_agent_batch.json)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="Add a single PR by number (skips merge-status filter, useful for unmerged PRs)",
    )
    args = parser.parse_args()

    mode_label = "PR-first" + (" + Agent" if args.agent else "")
    print(f"=== NPU Defect Mining — Daily Append [{mode_label}] ===")
    print(f"Time: {now_iso()}")

    db = ensure_defect_db()
    print(f"Current DB: {db['_meta']['total_defects']} defects")

    if args.pr:
        # Single PR mode: fetch one PR regardless of merge status
        new_defects = _scan_single_pr(args.pr, db)
    else:
        new_defects = scan_new_prs(db)
    if not new_defects:
        print("No new PRs found. Nothing to do.")
        return

    print(f"\nNew defects to add: {len(new_defects)}")
    for d in new_defects:
        print(f"  {d['bug_id']}: {d['pr_title'][:80]}...")

    if args.dry_run:
        print("\n[DRY RUN] Not saving changes.")
        return

    db = append_to_db(db, new_defects)
    save_json(DEFECT_DB_PATH, db)
    print(
        f"\nAdded {len(new_defects)} defects. DB now has {db['_meta']['total_defects']}."
    )

    if args.agent:
        from classify_pr import format_pr_batches as fmt_pr

        pr_defects = [d for d in new_defects if d.get("source") == "pr"]
        if pr_defects:
            pr_batches = fmt_pr(pr_defects, batch_size=min(len(pr_defects), 15))
            batch = {
                "generated_at": now_iso(),
                "mode": "incremental_pr_first",
                "pr_batches": pr_batches,
                "commit_batch": {"count": 0, "defects": []},
            }
            save_json(DB_DIR / "pending_agent_batch.json", batch)
            print(f"Agent batch prepared: {len(pr_defects)} defects")
            print("Next: Run Agent classification -> apply_agent_results.py")
    else:
        print("(No --agent; defects appended with category=unknown)")

    print(f"\nNext scan starts from: {db['_meta']['last_scan_commit'][:10]}")


if __name__ == "__main__":
    main()
