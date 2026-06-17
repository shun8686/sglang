#!/usr/bin/env python3
"""
Full Analysis Runner (M0 Baseline)

Performs a complete scan: feature extraction → Agent test coverage mapping →
risk profiling → dependency map generation.

Usage:
    python .claude/skills/npu-risk-graph/scripts/run_full.py

The script uses a two-step workflow for Phase 2 (test→feature mapping):
  1. First run: prepare Agent batch + prompt files, print Workflow instruction
  2. Second run (after Workflow completes): apply Agent results, continue to Phase 3/4
"""

import sys
from pathlib import Path

# Add this script's directory for shared imports (common.py, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BASELINE_DIR,
    DB_DIR,
    REPO_ROOT,
    ROOT,
    RiskLevel,
    compute_fingerprint,
    factor_score,
    get_current_commit,
    init_directories,
    load_json,
    now_iso,
    save_json,
)

# ============================================================
# Phase 1: Feature Extraction (simulated — production uses Agent)
# ============================================================

# Features are read from baseline JSON (single source of truth).
# To update: re-run the Agent Workflow or manually edit features.json.
# Lazy-loaded — features.json may not exist on first run (Phase 1 creates it).
_npu_features_cache: list[dict] | None = None


def _load_features() -> list[dict]:
    global _npu_features_cache
    if _npu_features_cache is not None:
        return _npu_features_cache
    data = load_json(BASELINE_DIR / "features.json")
    feats = data.get("features", [])
    if not feats:
        raise RuntimeError(
            "features.json is empty. Run the Agent Workflow or restore from backup."
        )
    _npu_features_cache = feats
    return feats


# ============================================================
# Phase 1: Extract Features
# ============================================================


def extract_features() -> list[dict]:
    """Load features from baseline JSON (Agent-generated or curated).

    Does NOT overwrite Agent-generated features with hardcoded list.
    Only falls back to hardcoded if baseline is missing or corrupted.
    """
    # Check if Agent-generated features exist
    agent_meta = load_json(BASELINE_DIR / ".agent_meta.json")
    if agent_meta.get("agent_generated"):
        features = load_json(BASELINE_DIR / "features.json").get("features", [])
        if len(features) >= 10:  # sanity check
            return features

    # Fallback: use features.json as source of truth
    features = []
    for f in _load_features():
        fingerprint = compute_fingerprint(f["source_files"])
        features.append({**f, "fingerprint": fingerprint, "last_modified": now_iso()})
    return features


# ============================================================
# Phase 2: Map Test Coverage (Agent Workflow)
# ============================================================


def map_test_coverage(
    features: list[dict], workflow_results: str | None = None
) -> list[dict] | None:
    """Phase 2 via Agent Workflow: prepare batch, wait, apply results.

    Two-step process:
      1. First run: prepare batch JSON + .txt prompt files, return None
      2. Second run: detect cached Agent results, apply them, return tests

    Args:
        features: list of feature dicts.
        workflow_results: path to Workflow output JSON (from --workflow-results).
            When provided, the file is copied to agent_baseline_results.json
            and applied immediately.

    Returns:
        list[dict] if Agent results are available and applied.
        None if batch was just prepared (waiting for Workflow).
    """
    from apply_agent_baseline import apply
    from baseline_prompts import build_batch_prompt
    from prepare_agent_baseline import prepare

    results_path = BASELINE_DIR / "agent_baseline_results.json"

    # Bridge: accept --workflow-results <path> to copy Workflow output into place
    if workflow_results:
        import shutil

        wf_path = Path(workflow_results)
        if wf_path.exists():
            shutil.copy2(wf_path, results_path)
            print(f"  -> Workflow results copied: {wf_path} → {results_path}")
        else:
            print(f"  [WARN] --workflow-results path not found: {wf_path}")

    # Check if Agent results already exist (resume from checkpoint)
    if results_path.exists():
        print("  -> Found cached Agent results, applying...")
        output = apply(results_path=results_path)
        return output.get("tests", [])

    # Prepare batch for Workflow
    print("  -> Preparing Agent batch...")
    batch_output = prepare(batch_size=7)

    # Generate .txt prompt files for the Workflow Agent to read
    prompts_dir = REPO_ROOT / ".sglang-risk" / "prompts" / "baseline_batches"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(batch_output["batches"]):
        prompt_text = build_batch_prompt(batch, batch_output["features"], i)
        batch_path = prompts_dir / f"batch_{i:02d}.txt"
        batch_path.write_text(prompt_text, encoding="utf-8")
    print(
        f"  -> Generated {len(batch_output['batches'])} batch prompt files: {prompts_dir}"
    )

    # Clean up stale batch files beyond current count
    total = len(batch_output["batches"])
    for stale in prompts_dir.glob("batch_*.txt"):
        try:
            idx = int(stale.stem.split("_")[1])
            if idx >= total:
                stale.unlink()
        except (ValueError, IndexError):
            pass

    workflow_cmd = (
        'Workflow({scriptPath: ".claude/skills/npu-risk-graph/workflows/baseline-mapping.js", args: {batchCount: '
        + str(total)
        + "}})"
    )
    print(f"""
  {'='*58}
   Agent batch prepared: {batch_output['total_batches']} batches, {batch_output['total_test_files']} test files
   Prompt files: .sglang-risk/prompts/baseline_batches/batch_00.txt ~ batch_{total-1:02d}.txt

   Next steps:
   1. Run the Workflow:
      {workflow_cmd}
   2. Re-run this script to apply results:
      python .claude/skills/npu-risk-graph/scripts/run_full.py
  {'='*58}
""")
    return None


# ============================================================
# Phase 4: Risk Profiling
# ============================================================


def load_defect_db() -> dict:
    """Load defect database for historical_defects scoring."""
    return load_json(DB_DIR / "defect_db.json")


_change_freq_cache: dict[str, int] = {}


def _compute_all_change_freqs(features: list[dict]) -> dict[str, int]:
    """Pre-compute time-decay change_freq (1-5) for all features.

    Uses ``git log --since=365.days --format=%ct`` to get Unix timestamps
    for every commit touching each feature's source files.  Applies a
    1/sqrt(days_ago) decay so that recent commits dominate, then linearly
    normalizes the summed heat across all features to the 1-5 scale.
    """
    import math
    import subprocess
    import time
    from collections import defaultdict

    now = time.time()
    heat_map: dict[str, float] = {}

    for f in features:
        source_files = f.get("source_files", [])
        if not source_files:
            heat_map[f["name"]] = 0.0
            continue

        try:
            result = subprocess.run(
                ["git", "log", "--since=365.days", "--format=%ct", "--"] + source_files,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=REPO_ROOT,
                timeout=15,
            )
            timestamps = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and line.isdigit():
                    timestamps.append(int(line))
        except Exception:
            heat_map[f["name"]] = 0.0
            continue

        if not timestamps:
            heat_map[f["name"]] = 0.0
            continue

        # Decay-weighted sum: each commit contributes 1/sqrt(days_ago)
        heat = sum(1.0 / math.sqrt(max(1, (now - ts) / 86400)) for ts in timestamps)
        heat_map[f["name"]] = heat

    # Linear normalization to 1-5
    heats = list(heat_map.values())
    h_min, h_max = min(heats), max(heats)
    if h_max <= h_min:
        h_max = h_min + 1  # avoid division by zero

    result = {}
    for name, heat in heat_map.items():
        normalized = 1 + 4 * (heat - h_min) / (h_max - h_min)
        result[name] = min(5, max(1, round(normalized)))

    global _change_freq_cache
    _change_freq_cache = result
    return result


def compute_risk_profile(feature: dict, tests: list[dict], defect_db: dict) -> dict:
    """Compute full risk profile for a single feature."""
    name = feature["name"]
    npu_participation = feature.get("npu_participation", "medium")

    # Features not supported on NPU have zero NPU risk.
    if npu_participation == "not_supported":
        return {
            "feature": name,
            "probability": {
                "total": 1,
                "max": 25,
                "factors": {
                    "code_complexity": factor_score(
                        1, rationale="NPU unsupported", source="npu_participation"
                    ),
                    "change_frequency": factor_score(
                        1, rationale="NPU unsupported", source="npu_participation"
                    ),
                    "historical_defects": factor_score(
                        0, rationale="NPU unsupported", source="npu_participation"
                    ),
                    "dependency_depth": factor_score(
                        1, rationale="NPU unsupported", source="npu_participation"
                    ),
                    "test_effectiveness": factor_score(
                        1,
                        rationale="NPU unsupported — no tests needed",
                        source="npu_participation",
                    ),
                },
            },
            "impact": {
                "total": 1,
                "max": 20,
                "factors": {
                    "user_exposure": factor_score(
                        1, rationale="Zero NPU users", source="npu_participation"
                    ),
                    "failure_mode": factor_score(
                        1, rationale="Cannot fail on NPU", source="npu_participation"
                    ),
                },
            },
            "composite_score": 2,
            "level": "low",
            "test_depth": "L3",
            "fingerprint": feature.get("fingerprint", ""),
            "last_updated": now_iso(),
        }

    # Find tests covering this feature
    feature_tests = [t for t in tests if name in t.get("features_tested", [])]

    # ----- Probability Factors (max 25) -----

    # code_complexity (1-5)
    complexity = feature.get("complexity", 1)

    # change_frequency (1-5): time-decay weighted commits over 365 days.
    # Recent changes weigh more: weight = 1 / sqrt(days_ago).
    # Pre-computed by _compute_all_change_freqs() before the Phase 3 loop.
    change_freq = _change_freq_cache.get(name, 3)

    # historical_defects (0-5, 0 = clean, 5 = ≥5 bugs)
    # Uses features_affected from defect_db.json (persisted by backfill_defect_features.py).
    hist_defects = 0
    defect_details = {}
    if defect_db:
        relevant = [
            d
            for d in defect_db.get("defects", [])
            if name in d.get("features_affected", [])
        ]
        hist_defects = min(5, len(relevant))
        if relevant:
            categories = {}
            for d in relevant:
                cat = d.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            defect_details = {
                "total_bugs": len(relevant),
                "categories": categories,
                "last_bug_date": max(d.get("date_fixed", "") for d in relevant),
            }

    # dependency_depth (1-5)
    dep_count = len(feature.get("dependent_features", []))
    dependency_depth = min(5, max(1, dep_count + 1))

    # test_effectiveness (1-5): inverted — 5=no tests, 1=excellent tests
    if not feature_tests:
        test_eff = 5  # no tests = highest risk
    else:
        max_quality = max(t.get("quality_score", 0) for t in feature_tests)
        test_eff = max(1, 5 - max_quality)

    # Weighted sum (max 25)
    probability = (
        complexity * 1.0
        + change_freq * 1.0
        + hist_defects * 1.0
        + dependency_depth * 0.6
        + test_eff * 1.4
    )
    prob_score = min(25, max(1, round(probability)))

    # ----- Impact Factors (max 20) -----

    # user_exposure (1-5): based on models_using from features.json.
    # Sentinel values like "All models on NPU" / "All models" mean universal exposure.
    models_using = feature.get("models_using", [])
    models_set = {str(m).lower() for m in models_using}
    if models_set & {"all models on npu", "all models", "all_npu_models"}:
        user_exposure = 5  # universal — every NPU model affected
    elif len(models_using) >= 5:
        user_exposure = 4  # broad — major model families
    elif len(models_using) >= 3:
        user_exposure = 3  # moderate — several families
    elif len(models_using) >= 1:
        user_exposure = 2  # limited — 1-2 families
    else:
        user_exposure = 1  # none listed (should not happen for supported features)

    # failure_mode (1-5): manual annotation only.
    #  5 = silent precision loss (user won't notice until outputs are wrong)
    #  4 = silent corruption / data loss (timeout, hang, wrong results without crash)
    #  3 = degraded output quality (accuracy drop, but not silent failure)
    #  2 = functional breakage (crash, error — user immediately sees)
    #  1 = cosmetic / non-functional (log spam, minor perf regression)
    failure_mode = feature.get("failure_mode", 3)
    if not isinstance(failure_mode, int) or not 1 <= failure_mode <= 5:
        failure_mode = 3

    impact_score = min(20, max(1, round(user_exposure * 0.8 + failure_mode * 1.0)))

    # NPU participation adjustments: features with weak or platform-agnostic
    # involvement have lower NPU-specific risk exposure.
    if npu_participation == "platform_agnostic":
        impact_score = max(1, round(impact_score * 0.5))
    elif npu_participation == "weak":
        impact_score = max(1, round(impact_score * 0.75))

    # Composite
    composite = prob_score + impact_score
    level = RiskLevel.from_score(composite)
    depth = RiskLevel.test_depth(level)

    return {
        "feature": name,
        "probability": {
            "total": prob_score,
            "max": 25,
            "factors": {
                "code_complexity": factor_score(
                    complexity,
                    rationale=f"Cyclomatic complexity estimate",
                    source="AST",
                ),
                "change_frequency": factor_score(
                    change_freq,
                    rationale=f"Time-decay weighted commits over 365d (1/sqrt days_ago)",
                    source="git",
                ),
                "historical_defects": factor_score(
                    hist_defects,
                    rationale=f"Past bugs: {defect_details.get('total_bugs', 0)}",
                    source="defect_db",
                    details=defect_details,
                ),
                "dependency_depth": factor_score(
                    dependency_depth,
                    rationale=f"Depends on {dep_count} other features",
                    source="features.json",
                ),
                "test_effectiveness": factor_score(
                    test_eff,
                    rationale=f"Best test quality: {max((t.get('quality_score',0) for t in feature_tests), default=0)}/5",
                    source="test_evaluator",
                ),
            },
        },
        "impact": {
            "total": impact_score,
            "max": 20,
            "factors": {
                "user_exposure": factor_score(
                    user_exposure,
                    rationale=f"Used by {len(feature.get('models_using',[]))} model families",
                    source="manual",
                ),
                "failure_mode": factor_score(
                    failure_mode,
                    rationale="Severity: what happens when it breaks",
                    source="manual",
                ),
            },
        },
        "composite_score": composite,
        "level": level,
        "test_depth": depth,
        "fingerprint": feature.get("fingerprint", ""),
        "last_updated": now_iso(),
    }


# ============================================================
# Phase 3: Dependency Map
# ============================================================


def build_dependency_map(features: list[dict], tests: list[dict]) -> dict:
    """Build file→feature mapping for incremental change detection."""
    dep_map = {}

    # Source files → features
    for f in features:
        for sf in f.get("source_files", []):
            sf_norm = sf.replace("\\", "/")
            if sf_norm not in dep_map:
                dep_map[sf_norm] = {
                    "features": [],
                    "risk_factors": [],
                    "file_type": "source",
                }
            dep_map[sf_norm]["features"].append(f["name"])
            dep_map[sf_norm]["risk_factors"].append("code_complexity")
            dep_map[sf_norm]["risk_factors"] = list(
                set(dep_map[sf_norm]["risk_factors"])
            )

    # Env var file → all features
    env_path = "python/sglang/srt/environ.py"
    dep_map[env_path] = {
        "features": [f["name"] for f in features if f.get("env_vars")],
        "risk_factors": ["change_frequency"],
        "file_type": "config",
    }

    # Server args → all features
    args_path = "python/sglang/srt/server_args.py"
    dep_map[args_path] = {
        "features": [f["name"] for f in features],
        "risk_factors": ["change_frequency"],
        "file_type": "config",
    }

    # Test files → features
    for t in tests:
        tf = t["test_file"]
        dep_map[tf] = {
            "features": t.get("features_tested", []),
            "risk_factors": ["test_effectiveness"],
            "file_type": "test",
            "is_test": True,
            "covers_files": t.get("covers_files", []),
            "coverage_ratio": t.get("coverage_ratio", 0.0),
        }

    return dep_map


# ============================================================
# Main
# ============================================================


def run_full(workflow_results: str | None = None):
    """Execute complete analysis pipeline.

    Args:
        workflow_results: path to Workflow output JSON. When provided, copies
            to agent_baseline_results.json and applies immediately (skips
            batch preparation if file exists).
    """
    init_directories()
    commit = get_current_commit()

    print(f"\n{'='*60}")
    print(f"NPU Risk Framework — Full Analysis")
    print(f"Commit: {commit}")
    print(f"{'='*60}\n")

    # Phase 1: Extract features
    print("[1/4] Extracting NPU features...")
    features = extract_features()
    # Always bump git_commit so delta commands diff against the correct baseline.
    # Agent-generated features are preserved; only metadata is updated.
    agent_meta = load_json(BASELINE_DIR / ".agent_meta.json")
    features_data = load_json(BASELINE_DIR / "features.json")
    features_data["git_commit"] = commit
    features_data["generated_at"] = now_iso()
    # Always update features with fingerprint + last_modified (from extract_features),
    # even when agent_generated=True (preserves all existing fields + adds derived ones)
    features_data["total_features"] = len(features)
    for f in features:
        # Merge fingerprint/last_modified into each feature without overwriting
        # agent-authored fields (description, complexity, failure_mode, etc.).
        # Compute fingerprint if missing (e.g. agent_generated features don't have it).
        fp = f.get("fingerprint") or compute_fingerprint(f.get("source_files", []))
        existing = next(
            (ef for ef in features_data.get("features", []) if ef["name"] == f["name"]),
            None,
        )
        if existing:
            existing["fingerprint"] = existing.get("fingerprint") or fp
            # Keep existing last_modified if already present and still valid
            if not existing.get("last_modified"):
                existing["last_modified"] = f.get("last_modified", now_iso())
        else:
            # New feature: ensure it has all required derived fields
            if not f.get("fingerprint"):
                f["fingerprint"] = fp
            if not f.get("last_modified"):
                f["last_modified"] = now_iso()
            features_data.setdefault("features", []).append(f)

    save_json(BASELINE_DIR / "features.json", features_data)
    print(
        f"  → {len(features)} features (source: {'agent' if agent_meta.get('agent_generated') else 'curated'})"
    )

    # Phase 2: Map test coverage (Agent Workflow, two-step)
    print("[2/4] Mapping test coverage (Agent Workflow)...")
    tests = map_test_coverage(features, workflow_results=workflow_results)
    if tests is None:
        return None  # Batch prepared — waiting for Workflow to complete

    save_json(
        BASELINE_DIR / "tests.json",
        {
            "generated_at": now_iso(),
            "git_commit": commit,
            "total_tests": len(tests),
            "tests": tests,
        },
    )
    # Summary stats
    total_methods = sum(len(t.get("test_methods", [])) for t in tests)
    quality_dist = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for t in tests:
        q = str(min(5, max(0, t.get("quality_score", 0))))
        quality_dist[q] = quality_dist.get(q, 0) + 1
    print(f"  → {len(tests)} test files, {total_methods} test methods")
    print(f"  → Quality distribution: {quality_dist}")

    # Phase 3: Dependency map (build BEFORE risk profiling so backfill can use it)
    print("[3/4] Building dependency map...")
    dep_map = build_dependency_map(features, tests)
    save_json(
        BASELINE_DIR / "dependency_map.json",
        {
            "generated_at": now_iso(),
            "git_commit": commit,
            "total_entries": len(dep_map),
            "dependency_map": dep_map,
        },
    )
    print(f"  → {len(dep_map)} file mappings")

    # Phase 4: Risk profiling
    print("[4/4] Computing risk profiles...")
    change_freqs = _compute_all_change_freqs(features)
    cf_dist = {}
    for v in change_freqs.values():
        cf_dist[v] = cf_dist.get(v, 0) + 1
    print(f"  → change_freq (time-decay): {dict(sorted(cf_dist.items()))}")
    defect_db = load_defect_db()

    # Ensure features_affected is persisted (runs backfill if needed)
    # dependency_map.json is now fresh, so file→feature reverse index is up-to-date
    feature_names = {f["name"] for f in features}
    needs_backfill = any(
        "features_affected" not in d
        or any(fn not in feature_names for fn in d.get("features_affected", []))
        for d in defect_db.get("defects", [])
    )
    if needs_backfill:
        missing = any(
            "features_affected" not in d for d in defect_db.get("defects", [])
        )
        reason = (
            "missing"
            if missing
            else "stale feature references (features.json was regenerated)"
        )
        print(f"  -> features_affected {reason}, running backfill...")
        from backfill_defect_features import backfill

        backfill()
        defect_db = load_defect_db()  # reload with persisted fields

        # Warn if Agent backfill is still needed (mechanical match was incomplete).
        # The risk_profiles.json produced in this run will use partial defect→feature
        # mappings until the user runs `/npu-risk-graph backfill` with Workflow results.
        still_unmatched = sum(
            1 for d in defect_db.get("defects", []) if not d.get("features_affected")
        )
        if still_unmatched > 0:
            print(
                f"  [WARN] {still_unmatched} defects still have no features_affected."
            )
            print(f"  [WARN] Risk scores use PARTIAL defect data.")
            print(
                f"  [WARN] Run `/npu-risk-graph backfill` with Agent Workflow, then re-run baseline."
            )

    profiles = []
    for f in features:
        profile = compute_risk_profile(f, tests, defect_db)
        profiles.append(profile)
    save_json(
        BASELINE_DIR / "risk_profiles.json",
        {
            "generated_at": now_iso(),
            "git_commit": commit,
            "profiles": profiles,
        },
    )

    # Summary
    by_level = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for p in profiles:
        by_level[p["level"]] = by_level.get(p["level"], 0) + 1

    print(f"  → Risk levels: {by_level}")
    print(f"  → Top risks:")
    for p in sorted(profiles, key=lambda x: x["composite_score"], reverse=True)[:5]:
        print(
            f"     {p['feature']:35s}  score={p['composite_score']:2d}/45  {p['level']}  depth={p['test_depth']}"
        )

    # Done
    print(f"\n{'='*60}")
    print(f"Baseline saved to: {BASELINE_DIR}")
    print(f"  features.json         — {len(features)} features")
    print(f"  tests.json            — {len(tests)} test files")
    print(f"  dependency_map.json   — {len(dep_map)} file mappings")
    print(f"  risk_profiles.json    — {len(profiles)} risk profiles")
    print(f"{'='*60}\n")

    return features, tests, profiles, dep_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NPU Risk Framework — Full Analysis")
    parser.add_argument(
        "--workflow-results",
        default=None,
        help="Path to Workflow output JSON (bridges 2-step Agent workflow)",
    )
    args = parser.parse_args()
    run_full(workflow_results=args.workflow_results)
