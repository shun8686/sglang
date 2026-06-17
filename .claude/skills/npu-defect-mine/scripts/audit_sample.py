#!/usr/bin/env python3
"""
Extract a stratified sample from defect_db.json for Agent Audit.

Strategy:
  - All needs_review defects
  - From each category: lowest 2 + highest 2 by confidence
  - "compatibility" gets 2x sampling (known catch-all category)
  - Prunes PR body to 2000 chars for prompt efficiency

Output: .sglang-risk/db/audit_batch.json (ready for Workflow agent() call)

Usage:
    python .claude/skills/npu-defect-mine/scripts/audit_sample.py [--sample-size 3]
"""

from common import DB_DIR, load_json, now_iso, save_json


def extract_sample(db: dict, per_category: int = 3) -> list[dict]:
    """Extract stratified audit sample from defect_db."""
    defects = db.get("defects", [])
    selected = set()
    sample = []

    # 1. All needs_review defects (highest priority)
    nr = [d for d in defects if d.get("needs_review")]
    for d in nr:
        selected.add(d["bug_id"])
        sample.append(d)
    if nr:
        print(f"  needs_review: {len(nr)}")

    # 2. Per-category stratified sampling (prioritize least-audited)
    by_cat = {}
    for d in defects:
        cat = d.get("category", "unknown")
        by_cat.setdefault(cat, []).append(d)

    for cat, cat_defects in sorted(by_cat.items()):
        # Sort by audit_count ASC, then confidence (low first) as tiebreaker
        # This ensures least-audited defects get sampled first
        sorted_defects = sorted(
            cat_defects, key=lambda d: (d.get("audit_count", 0), d.get("confidence", 0))
        )

        # "compatibility" gets extra sampling (known catch-all)
        n = per_category * 2 if cat == "compatibility" else per_category

        # Pick lowest audit_count (with low confidence preference)
        # and highest confidence (with low audit_count preference)
        picks = []
        for d in sorted_defects[:n]:
            if d["bug_id"] not in selected:
                picks.append(("low", d))
        # For high-confidence, sort by audit_count ASC, then confidence DESC
        high_sorted = sorted(
            cat_defects,
            key=lambda d: (d.get("audit_count", 0), -d.get("confidence", 0)),
        )
        for d in high_sorted[:n]:
            if d["bug_id"] not in selected:
                picks.append(("high", d))

        for reason, d in picks:
            if d["bug_id"] not in selected:
                selected.add(d["bug_id"])
                sample.append(
                    {
                        **d,
                        "_audit_reason": f"{cat}/{reason}_conf/audits={d.get('audit_count',0)}",
                    }
                )

        if picks:
            print(f"  {cat}: +{len(picks)} ({n*2} max)")

    print(f"\n  Total sample: {len(sample)} / {len(defects)} defects")
    return sample


def format_for_audit(defects: list[dict]) -> list[dict]:
    """Format defects into compact audit entries for the Agent prompt."""
    entries = []
    for d in defects:
        # Trim PR body for prompt efficiency
        body = (d.get("pr_body") or "")[:2000]
        title = d.get("pr_title") or d.get("title", "")

        entries.append(
            {
                "bug_id": d["bug_id"],
                "pr_title": title[:200],
                "pr_body": body,
                "pr_labels": d.get("pr_labels", [])[:5],
                "files_fixed": d.get("files_fixed", [])[:10],
                "current_category": d.get("category", "unknown"),
                "current_severity": d.get("severity", 0),
                "current_root_cause": d.get("root_cause", ""),
                "current_rationale": d.get("rationale", ""),
                "current_confidence": d.get("confidence", 0),
                "needs_review": d.get("needs_review", False),
                "_audit_reason": d.get("_audit_reason", ""),
            }
        )

    return entries


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract audit sample from defect_db")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Defects per category per confidence band (default: 3)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Audit ALL defects (override sample-size)"
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        metavar="N",
        help="Split output into per-batch files of N defects each (for parallel Agent audit)",
    )
    args = parser.parse_args()

    db = load_json(DB_DIR / "defect_db.json")

    if args.all:
        print(f"Extracting ALL {len(db.get('defects',[]))} defects for audit...")
        sample = [{"_audit_reason": "full_audit", **d} for d in db["defects"]]
    else:
        print(f"Extracting audit sample (per_category={args.sample_size})...")
        sample = extract_sample(db, per_category=args.sample_size)
    entries = format_for_audit(sample)

    audit_batch = {
        "generated_at": now_iso(),
        "mode": "audit",
        "total_sampled": len(entries),
        "total_in_db": len(db.get("defects", [])),
        "defects": entries,
    }

    output_path = DB_DIR / "audit_batch.json"
    save_json(output_path, audit_batch)
    print(f"\nAudit batch saved to: {output_path}")

    if args.split > 0:
        batch_dir = DB_DIR / "audit_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(entries), args.split):
            chunk = entries[i : i + args.split]
            batch_id = i // args.split
            save_json(
                batch_dir / f"audit_batch_{batch_id:02d}.json",
                {
                    "batch_id": batch_id,
                    "total_batches": (len(entries) + args.split - 1) // args.split,
                    "count": len(chunk),
                    "defects": chunk,
                },
            )
        print(
            f"Split into { (len(entries) + args.split - 1) // args.split } batch files in {batch_dir}/"
        )
        print(f"Next: Run Agent audit subagents (each reads one batch file)")


if __name__ == "__main__":
    main()
