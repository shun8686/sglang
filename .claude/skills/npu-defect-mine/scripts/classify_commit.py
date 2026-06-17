#!/usr/bin/env python3
"""
Format commit-only defects for Agent classification via Workflow.

Reads defect_db.json, extracts defects without PR data (source="commit") that
haven't been agent-classified yet, formats them using AGENT_COMMIT_PROMPT from
prompts.py, and prepares batch JSON for the Workflow's agent() calls.

Usage:
    python .claude/skills/npu-defect-mine/scripts/classify_commit.py [--all] [--output pending_agent_batch.json]

    Normal mode: formats commit defects needing Agent classification.
    --all mode: re-formats ALL commit defects regardless of existing agent_version.

    If a pending_agent_batch.json already exists from classify_pr.py,
    this script merges commit defects into it (rather than overwriting).
"""

from pathlib import Path

from common import DB_DIR, load_json, now_iso, save_json
from prompts import AGENT_COMMIT_PROMPT


def _has_agent_classification(defect: dict) -> bool:
    """Check if a defect has already been agent-classified."""
    av = defect.get("agent_version", "")
    skip_patterns = ("claude_semantic", "agent_workflow", "claude_final")
    return any(p in av for p in skip_patterns)


def build_commit_entry(defect: dict) -> dict:
    """Build a single formatted entry with all AGENT_COMMIT_PROMPT template variables."""
    diff_stats = defect.get("diff_stats", {})
    if isinstance(diff_stats, dict):
        diff_summary = diff_stats.get("summary", "")
        diff_files = diff_stats.get("files", [])
    else:
        diff_summary = str(diff_stats)[:200]
        diff_files = []

    return {
        "bug_id": defect["bug_id"],
        "sha": defect.get("commit_sha", "")[:10],
        "message": defect.get("title", "")[:500],
        "files_changed": "\n".join(
            diff_files if diff_files else defect.get("files_fixed", [])
        )[:1000]
        or "(no file data available)",
        "diff_stats": diff_summary[:200] or "(no diff stats available)",
    }


def render_prompt(entry: dict) -> str:
    """Render AGENT_COMMIT_PROMPT with an entry's variables filled in."""
    return AGENT_COMMIT_PROMPT.format(
        sha=entry["sha"],
        message=entry["message"],
        files_changed=entry["files_changed"],
        diff_stats=entry["diff_stats"],
    )


def format_commit_batch(defects: list[dict]) -> dict:
    """Format commit-only defects into a single batch.

    Commit-only defects are typically few in number and have limited data,
    so they're always grouped into one batch for a single agent() call.
    """
    if not defects:
        return {"count": 0, "defects": []}

    entries = [build_commit_entry(d) for d in defects]

    # Build combined prompt
    entries_text = []
    for j, entry in enumerate(entries):
        entries_text.append(f"""
--- Defect {j + 1}: {entry['bug_id']} ---
Commit SHA: {entry['sha']}
Commit Message: {entry['message']}
Files Changed: {entry['files_changed']}
Diff Stats: {entry['diff_stats']}
""")

    prompt_text = f"""You are an NPU defect classification expert working with LIMITED data.
Classify these commit-only defects (no Pull Request data available).

## Task
For each defect, provide your best classification based on the commit message and file changes:
1. Category: precision_loss | crash | perf_regression | compatibility | compile_error
2. Severity: 10=Critical (crash/OOM/hang), 3=Major (function error / precision loss), 1=Minor (perf / compile / boundary)
3. root_cause: specific one-sentence description, or "insufficient data — commit message only"
4. confidence: 0.50-0.70 maximum (commit data is inherently limited)
8. needs_review: true (ALL commit-only classifications require human review)
9. rationale: "Limited to commit message analysis. PR data not available."

## Important
- Be conservative: set confidence to 0.5-0.7 maximum
- Set needs_review: true for ALL entries
- Use "unknown" for any field you cannot determine from the commit message alone

## Defect Records
{''.join(entries_text)}

## Output Format
Return a JSON object with a "defects" array containing one classification per defect, in the same order."""

    return {
        "count": len(entries),
        "defects": entries,
        "prompt_text": prompt_text,
    }


def format_commit_single(defect: dict) -> dict:
    """Format a single commit defect for Phase B incremental classification."""
    return format_commit_batch([defect])


def load_commit_defects(force_all: bool = False) -> list[dict]:
    """Load commit-only defects from defect_db.json that need Agent classification."""
    db = load_json(DB_DIR / "defect_db.json")
    defects = db.get("defects", [])

    # Filter for commit-sourced defects (no PR data)
    commit_defects = [d for d in defects if d.get("source") == "commit"]

    if not force_all:
        commit_defects = [d for d in commit_defects if not _has_agent_classification(d)]

    return commit_defects


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Format commit defects for Agent classification"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-format ALL commit defects (even already classified ones)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: .sglang-risk/db/pending_agent_batch.json)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge into existing pending_agent_batch.json (from classify_pr.py)",
    )
    args = parser.parse_args()

    defects = load_commit_defects(force_all=args.all)

    if not defects:
        print("No commit-only defects found needing Agent classification.")
        if not args.merge:
            print("Nothing to output.")
        return

    commit_batch = format_commit_batch(defects)

    output_path = (
        Path(args.output) if args.output else (DB_DIR / "pending_agent_batch.json")
    )

    # If merging, load existing PR batch data
    if args.merge and output_path.exists():
        existing = load_json(output_path)
        pr_batches = existing.get("pr_batches", [])
        pr_count = sum(b.get("count", 0) for b in pr_batches)
    else:
        pr_batches = []
        pr_count = 0

    output_data = {
        "generated_at": now_iso(),
        "mode": "full" if pr_batches else "commit_only",
        "pr_batches": pr_batches,
        "commit_batch": commit_batch,
        "summary": {
            "total_pr_defects": pr_count,
            "total_commit_defects": len(defects),
            "pr_batches": len(pr_batches),
            "commit_batches": 1 if defects else 0,
        },
    }

    save_json(output_path, output_data)
    print(f"Formatted {len(defects)} commit-only defects into 1 batch")
    if pr_batches:
        print(
            f"Merged with {pr_count} PR defects in {len(pr_batches)} existing batches"
        )
    print(f"Output: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Verify the batch: cat {output_path} | python -m json.tool | head -50")
    print(f"  2. Run the Workflow: /workflow npu-defect-classify")
    print(
        f"  3. Run: python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py"
    )


if __name__ == "__main__":
    main()
