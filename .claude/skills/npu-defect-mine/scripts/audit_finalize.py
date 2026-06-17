#!/usr/bin/env python3
"""Increment audit_count for all defects found in audit result files.
Run after audit results are applied to defect_db.json."""

import glob
import json
from pathlib import Path

from common import DB_DIR, load_json, now_iso, save_json


def main():
    RESULT_GLOB = str(DB_DIR / "audit_batches" / "*result*")

    db = load_json(DB_DIR / "defect_db.json")

    # Collect all audited bug_ids from result files
    audited = set()
    result_mtimes = []
    for path in sorted(glob.glob(RESULT_GLOB)):
        result_mtimes.append(Path(path).stat().st_mtime)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["findings"]:
            audited.add(r["bug_id"])

    # ── Verify apply_agent_results.py was run ──
    last_apply = db["_meta"].get("agent_workflow_last_run", "")
    runs = db["_meta"].get("agent_workflow_runs", [])
    if not last_apply:
        print(
            "WARNING: agent_workflow_last_run is missing — apply_agent_results.py may not have been run!"
        )
        print("  Audit corrections may NOT have been applied to defect_db.json.")
        print(
            "  Run: python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py --input .sglang-risk/db/audit_report.json"
        )
    elif runs:
        last_run = runs[-1]
        if last_run.get("updated", 0) == 0:
            print(
                "WARNING: Last apply_agent_results run updated 0 defects — audit corrections were NOT applied!"
            )
            print(
                "  Possible format error in audit_report.json. Check the apply output above."
            )

    # Increment
    for d in db["defects"]:
        if d["bug_id"] in audited:
            d["audit_count"] = d.get("audit_count", 0) + 1

    db["_meta"]["last_audit_at"] = now_iso()
    save_json(DB_DIR / "defect_db.json", db)

    # Report
    by_count = {}
    for d in db["defects"]:
        c = d.get("audit_count", 0)
        by_count[c] = by_count.get(c, 0) + 1

    print(f"Incremented audit_count for {len(audited)} defects")
    print(f"Distribution: {dict(sorted(by_count.items()))}")

    # Check for defects in result files not found in DB
    db_ids = {d["bug_id"] for d in db["defects"]}
    orphan_results = audited - db_ids
    if orphan_results:
        print(
            f"WARNING: {len(orphan_results)} result bug_ids not in DB: {orphan_results}"
        )


if __name__ == "__main__":
    main()
