#!/usr/bin/env python3
"""
Backfill features_affected into defect_db.json.

Two-step process:
  1. Mechanical file->feature matching (fast). For unmatched defects, prepare
     Agent batch prompts and print a Workflow instruction.
  2. Re-run after Workflow completes — applies Agent results to unmatched defects.

Run:
    python .claude/skills/npu-risk-graph/scripts/backfill_defect_features.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Add this script's directory for shared imports (common.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BASELINE_DIR,
    DB_DIR,
    REPO_ROOT,
    load_json,
    now_iso,
    save_json,
)

# ---------------------------------------------------------------------------
# Mechanical matching (same as before)
# ---------------------------------------------------------------------------

CATEGORY_TO_FAILURE_MODE = {
    "crash": "crash",
    "precision_loss": "silent_precision_loss",
    "compatibility": "compatibility_issue",
    "perf_regression": "performance_regression",
    "compile_error": "build_error",
}

CATEGORY_IS_SILENT = {
    "precision_loss": True,
    "perf_regression": True,
    "crash": False,
    "compile_error": False,
    "compatibility": False,
}


def _build_reverse_index(features, dep_map):
    """Build file -> {feature_names} reverse index."""
    file_to_features: dict[str, set[str]] = {}
    for file_path, entry in dep_map.items():
        file_to_features.setdefault(file_path, set()).update(entry.get("features", []))
    for f in features:
        for sf in f.get("source_files", []):
            file_to_features.setdefault(sf.replace("\\", "/"), set()).add(f["name"])
    return file_to_features


def _mechanical_match(defects, file_to_features, hub_files):
    """Mechanically match defects to features. Returns (matched_count, zero_matched_defects)."""
    zero_matched = []
    matched = 0
    for d in defects:
        features_affected = set()
        for fp in d.get("files_fixed", []):
            fp_norm = fp.replace("\\", "/")
            if fp_norm in hub_files:
                continue
            if fp_norm in file_to_features:
                features_affected.update(file_to_features[fp_norm])
        d["features_affected"] = sorted(features_affected)
        if features_affected:
            matched += 1
        else:
            zero_matched.append(d)
    return matched, zero_matched


# ---------------------------------------------------------------------------
# Agent batch preparation
# ---------------------------------------------------------------------------


def _prepare_agent_batch(zero_defects, features):
    """Write prompt files for Agent to map unmatched defects to features."""
    prompts_dir = REPO_ROOT / ".sglang-risk" / "prompts" / "defect_backfill_batches"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 5
    batches = [
        zero_defects[i : i + batch_size]
        for i in range(0, len(zero_defects), batch_size)
    ]

    feature_list = "\n".join(
        f"- {f['name']} ({f.get('category','?')}): {f.get('description','')[:100]}"
        for f in sorted(features, key=lambda x: x["name"])
    )

    global_idx = 0
    for bi, batch in enumerate(batches):
        defect_lines = []
        for d in batch:
            desc = d.get("description", "") or d.get("title", "") or "(no description)"
            files = d.get("files_fixed", [])
            cat = d.get("category", "unknown")
            defect_lines.append(
                f"### Defect {global_idx}\n"
                f"- Category: {cat}\n"
                f"- Description: {desc}\n"
                f"- Files fixed: {', '.join(files[:5])}\n"
            )
            global_idx += 1

        prompt = f"""You are an NPU defect analyst. Map each defect below to the features it affects.

## Available Features

{feature_list}

## Defects to Map

{''.join(defect_lines)}

## Instructions

For each defect, determine which features from the list above are affected.
A defect may affect 0-5 features. Consider:
- Which feature's code is in the files_fixed?
- What kind of bug is it (category)?
- The description may reference specific components/APIs that belong to features.

## Output

Return a JSON object:
{{"mappings": [
  {{"defect_index": 0, "features_affected": ["feature_a", "feature_b"]}},
  ...
]}}

Return ONLY valid JSON — no preamble, no markdown fences."""
        batch_path = prompts_dir / f"batch_{bi:02d}.txt"
        batch_path.write_text(prompt, encoding="utf-8")

    return len(batches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def backfill(workflow_results: str | None = None):
    """Backfill features_affected into defect_db.json.

    Args:
        workflow_results: path to Workflow output JSON. When provided, copies
            to agent_defect_backfill_results.json and applies immediately.
    """
    # --- load data -------------------------------------------------
    defect_db = load_json(DB_DIR / "defect_db.json")
    dep_map_data = load_json(BASELINE_DIR / "dependency_map.json")
    features_data = load_json(BASELINE_DIR / "features.json")

    defects = defect_db.get("defects", [])
    dep_map = dep_map_data.get("dependency_map", {})
    features = features_data.get("features", [])

    # --- file->features reverse index --------------------------------
    file_to_features = _build_reverse_index(features, dep_map)

    # --- hub file exclusion ------------------------------------------
    total_features = len(features)
    hub_threshold = max(5, int(total_features * 0.25))
    hub_files = {
        fp for fp, feats in file_to_features.items() if len(feats) > hub_threshold
    }

    # --- Step 1: mechanical matching ---------------------------------
    matched, zero_defects = _mechanical_match(defects, file_to_features, hub_files)

    # --- backfill failure_mode + is_silent for ALL defects -----------
    for d in defects:
        cat = d.get("category", "unknown")
        if "failure_mode" not in d:
            d["failure_mode"] = CATEGORY_TO_FAILURE_MODE.get(cat, cat)
        if "is_silent" not in d:
            d["is_silent"] = CATEGORY_IS_SILENT.get(cat, False)

    # --- Step 2: Agent backfill for zero-matched defects -------------
    agent_results_path = BASELINE_DIR / "agent_defect_backfill_results.json"

    # Bridge: accept --workflow-results <path> to copy Workflow output into place
    if workflow_results:
        import shutil

        wf_path = Path(workflow_results)
        if wf_path.exists():
            shutil.copy2(wf_path, agent_results_path)
            print(f"  -> Workflow results copied: {wf_path} → {agent_results_path}")
        else:
            print(f"  [WARN] --workflow-results path not found: {wf_path}")

    if zero_defects and not agent_results_path.exists():
        # Prepare Agent batch
        num_batches = _prepare_agent_batch(zero_defects, features)
        print(f"Mechanical match: {matched}/{len(defects)}")
        print(f"Unmatched (need Agent): {len(zero_defects)}")
        print(f"  -> Generated {num_batches} batch prompt files")
        print()
        print(f"  {'='*58}")
        print(f"   Next steps:")
        print(f"   1. Run the Workflow to map unmapped defects:")
        print(
            f'      Workflow({{scriptPath: ".claude/skills/npu-risk-graph/workflows/defect-backfill.js",'
        )
        print(f"                args: {{batchCount: {num_batches}}}}})")
        print(f"   2. Re-run to apply results:")
        print(
            f"      python .claude/skills/npu-risk-graph/scripts/backfill_defect_features.py"
        )
        print(f"  {'='*58}")
        print()
        # Save partial results (mechanical matches only), don't update _meta stats yet
        backup_path = DB_DIR / "defect_db.json.bak"
        save_json(backup_path, load_json(DB_DIR / "defect_db.json"))
        save_json(DB_DIR / "defect_db.json", defect_db)
        print(f"Mechanical matches saved. Re-run after Workflow to complete.")
        return

    if zero_defects and agent_results_path.exists():
        print("  -> Found cached Agent defect mapping results, applying...")
        agent_data = load_json(agent_results_path)
        agent_mappings = agent_data.get("mappings", [])
        # Build lookup: defect_index -> features_affected
        agent_map = {
            m["defect_index"]: m.get("features_affected", []) for m in agent_mappings
        }
        agent_filled = 0
        for di, d in enumerate(zero_defects):
            if di in agent_map:
                d["features_affected"] = sorted(set(agent_map[di]))
                if d["features_affected"]:
                    agent_filled += 1
                    matched += 1
        print(f"  Agent filled: {agent_filled}/{len(zero_defects)}")

    # --- stats -------------------------------------------------------
    stats = Counter()
    for d in defects:
        n = len(d.get("features_affected", []))
        if n == 0:
            stats["zero"] += 1
        elif n <= 3:
            stats["1-3"] += 1
        elif n <= 8:
            stats["4-8"] += 1
        else:
            stats["9+"] += 1

    # --- update metadata ---------------------------------------------
    defect_db["_meta"]["features_affected_backfilled_at"] = now_iso()
    defect_db["_meta"]["features_affected_hub_files"] = sorted(hub_files)
    defect_db["_meta"]["features_affected_stats"] = dict(stats)

    # --- save --------------------------------------------------------
    backup_path = DB_DIR / "defect_db.json.bak"
    save_json(backup_path, load_json(DB_DIR / "defect_db.json"))
    save_json(DB_DIR / "defect_db.json", defect_db)

    print(f"Backfill complete.")
    print(
        f"  Defects with features_affected: {len(defects) - stats['zero']}/{len(defects)}"
    )
    print(f"  Hub files excluded: {len(hub_files)}")
    print(f"  Distribution: {dict(stats)}")
    if agent_results_path.exists():
        print(f"  Agent results applied from: {agent_results_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill features_affected into defect_db.json"
    )
    parser.add_argument(
        "--workflow-results",
        default=None,
        help="Path to Workflow output JSON (bridges 2-step Agent workflow)",
    )
    args = parser.parse_args()
    backfill(workflow_results=args.workflow_results)
