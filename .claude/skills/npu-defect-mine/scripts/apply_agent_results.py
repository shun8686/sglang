#!/usr/bin/env python3
"""
Apply Agent classification results from Workflow output to defect_db.json.

Validates each classification against defect_db_schema.json before merging,
preserves non-classification fields (commit_sha, date_fixed, title, etc.),
and updates _meta tracking fields.

Usage:
    python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py [--input agent_results.json] [--dry-run]

    --input: Path to the Workflow output JSON (default: .sglang-risk/db/agent_results.json)
    --dry-run: Validate and print what would change without actually saving
"""

import json
import re
import sys
from pathlib import Path

from common import DB_DIR, load_json, now_iso, save_json

# Fields that the Agent produces (classification)
AGENT_FIELDS = {
    "category",
    "severity",
    "root_cause",
    "confidence",
    "needs_review",
    "rationale",
}

# Allowed enum values (mirrors defect_db_schema.json)
ALLOWED_CATEGORIES = {
    "precision_loss",
    "crash",
    "perf_regression",
    "compatibility",
    "compile_error",
}
ALLOWED_SEVERITIES = {1, 3, 10}  # Minor, Major, Critical


def _validate_bug_id(bug_id: str) -> bool:
    """Check bug_id format: BUG-YYYY-NNN"""
    return bool(re.match(r"^BUG-\d{4}-\d{3}$", bug_id))


def validate_classification(result: dict) -> tuple[list[str], list[str]]:
    """Validate a single classification result.

    Returns (errors, warnings) tuple.
    - errors: blocking issues (missing required fields, invalid enum values)
    - warnings: non-blocking issues (unknown feature names, etc.)
    """
    errors = []
    warnings = []

    # Required fields
    bug_id = result.get("bug_id", "")
    if not bug_id:
        errors.append("Missing required field: bug_id")
    elif not _validate_bug_id(bug_id):
        errors.append(f"Invalid bug_id format: {bug_id} (expected BUG-YYYY-NNN)")

    if "category" not in result:
        errors.append("Missing required field: category")
    elif result["category"] not in ALLOWED_CATEGORIES:
        errors.append(f"Invalid category: {result['category']}")

    if "severity" not in result:
        errors.append("Missing required field: severity")
    elif result["severity"] not in ALLOWED_SEVERITIES:
        errors.append(
            f"Invalid severity: {result['severity']} (expected one of {ALLOWED_SEVERITIES})"
        )

    if "root_cause" not in result:
        errors.append("Missing required field: root_cause")

    if "confidence" not in result:
        errors.append("Missing required field: confidence")
    elif not isinstance(result["confidence"], (int, float)) or not (
        0 <= result["confidence"] <= 1
    ):
        errors.append(f"Invalid confidence: {result['confidence']} (expected 0-1)")

    # Optional field validation (warn only, don't block)
    # failure_mode removed — category alone determines manifestation

    # root_cause is free-form text — no enum validation

    return errors, warnings


def apply_results(db: dict, results: list[dict]) -> dict:
    """Apply Workflow classification results to defect_db.

    Returns stats dict: {applied, updated, validation_errors, not_found}
    """
    defect_map = {d["bug_id"]: d for d in db.get("defects", [])}

    stats = {
        "applied": 0,
        "updated": 0,
        "validation_errors": 0,
        "not_found": 0,
        "details": [],
    }

    for r in results:
        bug_id = r.get("bug_id", "")

        if bug_id not in defect_map:
            stats["not_found"] += 1
            stats["details"].append(f"{bug_id}: not found in defect_db")
            continue

        errors, warnings = validate_classification(r)
        if errors:
            stats["validation_errors"] += 1
            stats["details"].append(f"{bug_id}: BLOCKED: {'; '.join(errors)}")
            continue
        if warnings:
            stats["details"].append(f"{bug_id}: warnings: {'; '.join(warnings)}")

        d = defect_map[bug_id]

        # Apply classification fields
        for field in AGENT_FIELDS:
            if field in r:
                d[field] = r[field]

        # Set agent metadata
        d["agent_version"] = "agent_workflow_v1"
        d["source"] = d.get("source", "pr")  # Preserve original source

        # Ensure needs_review is consistent with confidence
        if d.get("confidence", 0) < 0.7:
            d["needs_review"] = True

        stats["updated"] += 1
        stats["applied"] += 1

    # Update _meta
    meta = db.setdefault("_meta", {})
    runs = meta.setdefault("agent_workflow_runs", [])
    runs.append(
        {
            "run_at": now_iso(),
            "results_applied": stats["applied"],
            "updated": stats["updated"],
            "validation_errors": stats["validation_errors"],
            "not_found": stats["not_found"],
        }
    )
    meta["agent_workflow_last_run"] = now_iso()
    meta["agent_workflow_total_classified"] = (
        meta.get("agent_workflow_total_classified", 0) + stats["applied"]
    )

    return stats


def convert_audit_findings(findings: list[dict], db: dict) -> list[dict]:
    """Convert audit findings (format A) to classification records (format B).

    Audit format: {findings: [{bug_id, action, evidence, changes, confidence}]}
    Classification format: [{bug_id, category, severity, root_cause, confidence, rationale}]

    For action=accept: copies current DB values, clears needs_review.
    For action=change: merges changes{} over current DB values, clears needs_review.
    For action=delete: skips (non-defect, no classification to apply).
    """
    defect_map = {d["bug_id"]: d for d in db.get("defects", [])}
    results = []
    skipped = 0

    for f in findings:
        bug_id = f.get("bug_id", "")
        if bug_id not in defect_map:
            print(f"  WARNING: {bug_id} from audit not found in defect_db — skipping")
            skipped += 1
            continue

        d = defect_map[bug_id]
        action = f.get("action", "accept")

        if action == "delete":
            skipped += 1
            continue

        if action == "change":
            changes = f.get("changes", {})
            # is_not_npu flag: keep current classification, force needs_review for main Agent review
            suspect_npu = (
                changes.pop("is_not_npu", False) if isinstance(changes, dict) else False
            )
            results.append(
                {
                    "bug_id": bug_id,
                    "category": changes.get("category", d.get("category", "unknown")),
                    "severity": changes.get("severity", d.get("severity", 0)),
                    "root_cause": changes.get("root_cause", d.get("root_cause", "")),
                    "confidence": changes.get(
                        "confidence", f.get("confidence", d.get("confidence", 0))
                    ),
                    "rationale": changes.get("rationale", d.get("rationale", "")),
                    "needs_review": suspect_npu,  # main Agent must confirm deletion
                }
            )
        else:
            # action=accept: keep current values, audit confirmed → clear needs_review
            results.append(
                {
                    "bug_id": bug_id,
                    "category": d.get("category", "unknown"),
                    "severity": d.get("severity", 0),
                    "root_cause": d.get("root_cause", ""),
                    "confidence": d.get("confidence", 0),
                    "rationale": d.get("rationale", ""),
                    "needs_review": False,
                }
            )

    if skipped:
        print(f"  → {skipped} findings skipped (delete/not-found)")
    return results


def load_results(input_path: Path, db: dict = None) -> list[dict]:
    """Load classification results from Workflow output or audit report.

    Handles formats:
    - Single {defects: [...]} — Agent classification output
    - Array of {defects: [...]} — merged pipeline output
    - {findings: [...]} — audit report (auto-converts via convert_audit_findings)
    """
    data = load_json(input_path)
    if not data:
        print("ERROR: No data in results file")
        return []

    # Handle array of batch results
    if isinstance(data, list):
        all_defects = []
        for item in data:
            if isinstance(item, dict) and "defects" in item:
                all_defects.extend(item["defects"])
            elif isinstance(item, dict) and "bug_id" in item:
                all_defects.append(item)
        return all_defects

    # Handle audit format: {findings: [{action, evidence, changes}, ...]}
    if "findings" in data:
        if db is None:
            print(
                "ERROR: Audit format (findings) detected but no defect_db provided for conversion."
            )
            print(
                "This is a bug — load_results() must be called with db when processing audit reports."
            )
            return []
        print(
            f"  → Detected audit format (findings), converting to classification format..."
        )
        results = convert_audit_findings(data["findings"], db)
        print(f"  → Converted {len(results)} audit findings to classification records")
        return results

    # Handle single classification result
    if "defects" in data:
        return data["defects"]

    # Handle flat dict
    if isinstance(data, dict) and "bug_id" in data:
        return [data]

    print(
        "WARNING: Unexpected results format. Expected {defects: [...]}, {findings: [...]}, or [...]"
    )
    return []


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply Agent classification results to defect_db.json"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to agent results JSON (default: .sglang-risk/db/agent_results.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview changes without saving",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else (DB_DIR / "agent_results.json")

    if not input_path.exists():
        print(f"ERROR: Results file not found: {input_path}")
        print("Run the Workflow first to generate agent_results.json")
        sys.exit(1)

    # Load DB first (needed for audit format conversion + normal apply)
    print(f"\nLoading defect_db.json...")
    db = load_json(DB_DIR / "defect_db.json")
    total_before = len(db.get("defects", []))

    print(f"Loading results from: {input_path}")
    results = load_results(input_path, db)
    print(f"  → {len(results)} classification results loaded")

    if not results:
        print("No results to apply. Exiting.")
        return

    # Preview results
    print(f"\nClassification preview:")
    for r in results[:10]:
        print(
            f"  {r.get('bug_id', '??')}: cat={r.get('category','?')} "
            f"sev={r.get('severity','?')} "
            f"root={r.get('root_cause','?')} conf={r.get('confidence','?')}"
        )
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")

    stats = apply_results(db, results)

    # Print stats
    print(f"\nResults:")
    print(f"  Applied:     {stats['applied']}")
    print(f"  Updated:     {stats['updated']}")
    print(f"  Validation errors: {stats['validation_errors']}")
    print(f"  Not found:   {stats['not_found']}")

    if stats["details"]:
        print(f"\nDetails:")
        for detail in stats["details"]:
            print(f"  {detail}")

    # Confidence distribution after merge
    conf_dist = {"high(>=0.9)": 0, "med(0.7-0.9)": 0, "low(<0.7)": 0}
    for d in db["defects"]:
        c = d.get("confidence", 0)
        if c >= 0.9:
            conf_dist["high(>=0.9)"] += 1
        elif c >= 0.7:
            conf_dist["med(0.7-0.9)"] += 1
        else:
            conf_dist["low(<0.7)"] += 1
    print(f"\nConfidence distribution after merge: {conf_dist}")

    # Agent version distribution
    av_dist = {}
    for d in db["defects"]:
        av = d.get("agent_version", "unknown")
        av_dist[av] = av_dist.get(av, 0) + 1
    print(f"Agent version distribution: {av_dist}")

    if args.dry_run:
        print(f"\n[DRY RUN] Not saving. {stats['updated']} defects would be updated.")
    else:
        save_json(DB_DIR / "defect_db.json", db)
        print(
            f"\nSaved defect_db.json ({total_before} total, {stats['updated']} updated)"
        )
        print(f"Agent workflow run recorded in _meta.agent_workflow_runs")


if __name__ == "__main__":
    main()
