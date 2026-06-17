#!/usr/bin/env python3
"""Print defect_db.json status. Read-only, no modifications.

Usage:
    python .claude/skills/npu-defect-mine/scripts/status_check.py
"""

from common import DB_DIR, load_json


def main():
    db = load_json(DB_DIR / "defect_db.json")
    defects = db.get("defects", [])

    nr = sum(1 for d in defects if d.get("needs_review"))
    lc = sum(1 for d in defects if d.get("confidence", 0) < 0.7)
    hw = sum(1 for d in defects if d.get("confidence", 0) >= 0.9)

    av = {}
    cats = {}
    for d in defects:
        av[d.get("agent_version", "?")] = av.get(d.get("agent_version", "?"), 0) + 1
        cats[d.get("category", "?")] = cats.get(d.get("category", "?"), 0) + 1

    print(f"Total: {len(defects)} | needs_review: {nr} | low_conf: {lc}")
    print("Agent version:")
    for k, v in sorted(av.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print()
    print("Category:")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print()
    print(f"Confidence: high(>=0.9)={hw}, med={len(defects) - hw - lc}, low={lc}")

    meta = db.get("_meta", {})
    runs = meta.get("agent_workflow_runs", [])
    if runs:
        last = runs[-1]
        print(f"\nLast workflow: {last['run_at']} ({last['updated']} updated)")
    print(f"Extraction: {meta.get('extraction_method', 'unknown')}")


if __name__ == "__main__":
    main()
