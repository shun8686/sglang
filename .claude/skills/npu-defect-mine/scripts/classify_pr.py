#!/usr/bin/env python3
"""
Format PR-enriched defects for Agent classification via Workflow.

Reads defect_db.json, extracts defects with PR data that haven't been
agent-classified yet, formats them using AGENT_PR_PROMPT from prompts.py,
and prepares batch JSON for the Workflow's agent() calls.

Usage:
    python .claude/skills/npu-defect-mine/scripts/classify_pr.py [--all] [--batch-size 15] [--output pending_agent_batch.json]

    Normal mode: formats PR defects needing Agent classification (skips already-done).
    --all mode: re-formats ALL PR defects regardless of existing agent_version.
"""

from pathlib import Path

from common import DB_DIR, load_json, now_iso, save_json
from prompts import AGENT_PR_PROMPT


def _has_agent_classification(defect: dict) -> bool:
    """Check if a defect has already been agent-classified (skip it)."""
    av = defect.get("agent_version", "")
    skip_patterns = ("claude_semantic", "agent_workflow", "claude_final")
    return any(p in av for p in skip_patterns)


def _format_reviews(review_list: list[dict]) -> str:
    """Format review comments into a compact string for the prompt."""
    if not review_list:
        return "(no review comments available)"
    lines = []
    for r in review_list[:8]:  # Cap at 8 reviews to keep prompt size reasonable
        user = r.get("user", "unknown")
        body = r.get("body", "")[:300]
        path = r.get("path", "")
        lines.append(f"[{user} on {path}]: {body}")
    return "\n".join(lines)


def _extract_linked_issues(title: str) -> str:
    """Extract linked issue/PR references from commit title."""
    refs = []
    import re

    # Match #NNNNN patterns
    for m in re.finditer(r"#(\d{4,6})", title):
        refs.append(m.group(0))
    return ", ".join(refs) if refs else "(none)"


def build_pr_entry(defect: dict) -> dict:
    """Build a single formatted entry with all AGENT_PR_PROMPT template variables.

    Returns a dict that can be rendered into the prompt template.
    """
    pr_body = (defect.get("pr_body", "") or defect.get("description", "") or "")[:2000]

    reviews = defect.get("agent_reviews", []) or defect.get("reviews", [])
    review_text = _format_reviews(reviews)

    return {
        "bug_id": defect["bug_id"],
        "pr_title": defect.get("pr_title", "") or defect.get("title", "")[:200],
        "pr_body": pr_body,
        "pr_labels": defect.get("pr_labels", []),
        "review_comments": review_text,
        "linked_issues": _extract_linked_issues(defect.get("title", "")),
        "files_changed": defect.get("files_fixed", [])[:15],  # Cap files
    }


def render_prompt(entry: dict) -> str:
    """Render AGENT_PR_PROMPT with an entry's variables filled in."""
    return AGENT_PR_PROMPT.format(
        pr_title=entry["pr_title"],
        pr_body=entry["pr_body"],
        pr_labels=", ".join(entry["pr_labels"]) if entry["pr_labels"] else "(none)",
        review_comments=entry["review_comments"],
        linked_issues=entry["linked_issues"],
        files_changed=(
            "\n".join(entry["files_changed"]) if entry["files_changed"] else "(none)"
        ),
    )


def format_pr_batches(defects: list[dict], batch_size: int = 15) -> list[dict]:
    """Split defects into batches for parallel agent() calls.

    Each batch is self-contained with formatted entries and a pre-rendered
    prompt text ready for the Workflow agent() call.
    """
    batches = []
    for i in range(0, len(defects), batch_size):
        batch_defects = defects[i : i + batch_size]
        entries = [build_pr_entry(d) for d in batch_defects]

        # Build a combined prompt for the entire batch
        entries_text = []
        for j, entry in enumerate(entries):
            entries_text.append(f"""
--- Defect {j + 1}: {entry['bug_id']} ---
PR Title: {entry['pr_title']}
PR Description: {entry['pr_body']}
PR Labels: {', '.join(entry['pr_labels']) if entry['pr_labels'] else '(none)'}
Review Comments: {entry['review_comments']}
Linked Issues: {entry['linked_issues']}
Files Changed: {', '.join(entry['files_changed']) if entry['files_changed'] else '(none)'}
""")

        prompt_text = f"""You are an NPU (Huawei Ascend) defect classification expert.
Analyze the following {len(entries)} GitHub Pull Request defect records and classify each one.

## Task
For each defect, classify:
1. Category: precision_loss | crash | perf_regression | compatibility | compile_error
2. Severity: 10=Critical (crash/OOM/hang), 3=Major (function error / precision loss), 1=Minor (perf / compile / boundary)
3. root_cause: specific one-sentence description of the root cause (be concrete, not generic like "cann_api")
4. confidence (0.80-0.95 for PR-based analysis)
8. needs_review: true only if confidence < 0.7
9. rationale: one sentence explaining classification basis

## Defect Records
{''.join(entries_text)}

## Output Format
Return a JSON object with a "defects" array containing one classification per defect, in the same order.
Set agent classification confidence high (0.80-0.95) when PR data clearly indicates the defect type.
Only mark needs_review: true if the evidence is truly ambiguous."""

        batches.append(
            {
                "batch_id": len(batches),
                "count": len(entries),
                "defects": entries,  # Structured data for the Workflow
                "prompt_text": prompt_text,  # Pre-rendered prompt for convenience
            }
        )

    return batches


def format_pr_single(defect: dict) -> dict:
    """Format a single defect for Phase B incremental classification."""
    return format_pr_batches([defect], batch_size=1)[0]


def load_pr_defects(force_all: bool = False) -> list[dict]:
    """Load PR-enriched defects from defect_db.json that need Agent classification.

    Args:
        force_all: If True, include ALL PR defects regardless of agent_version.
                   If False, skip defects that have already been agent-classified.
    """
    db = load_json(DB_DIR / "defect_db.json")
    defects = db.get("defects", [])

    # Filter for PR-sourced defects
    pr_defects = [d for d in defects if d.get("source") == "pr"]

    if not force_all:
        pr_defects = [d for d in pr_defects if not _has_agent_classification(d)]

    return pr_defects


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Format PR defects for Agent classification"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-format ALL PR defects (even already classified ones)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=15, help="Defects per batch (default: 15)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: .sglang-risk/db/pending_agent_batch.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit to first N defects (for testing)"
    )
    args = parser.parse_args()

    defects = load_pr_defects(force_all=args.all)
    if args.limit > 0:
        defects = defects[: args.limit]

    if not defects:
        print("No PR defects found needing Agent classification.")
        print(
            "(All PR defects may already have agent classification, or no PR defects exist.)"
        )
        print("Use --all to re-classify everything.")
        return

    batches = format_pr_batches(defects, batch_size=args.batch_size)

    output_path = (
        Path(args.output) if args.output else (DB_DIR / "pending_agent_batch.json")
    )

    # Also include commit batch placeholder (classify_commit.py fills this in)
    output_data = {
        "generated_at": now_iso(),
        "mode": "pr_only",
        "pr_batches": batches,
        "commit_batch": {"count": 0, "defects": []},
        "summary": {
            "total_pr_defects": len(defects),
            "total_batches": len(batches),
            "batch_size": args.batch_size,
        },
    }

    save_json(output_path, output_data)
    print(
        f"Formatted {len(defects)} PR defects into {len(batches)} batches (size={args.batch_size})"
    )
    print(f"Output: {output_path}")
    print(f"\nNext steps:")
    print(
        f"  1. (Optional) Run classify_commit.py to add commit-only defects to the batch"
    )
    print(f"  2. Run the Workflow: /workflow npu-defect-classify")
    print(f"     (Claude will read pending_agent_batch.json and pass it as args)")
    print(
        f"  3. Run: python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py"
    )


if __name__ == "__main__":
    main()
