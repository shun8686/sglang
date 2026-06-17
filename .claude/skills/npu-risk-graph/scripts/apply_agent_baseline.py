#!/usr/bin/env python3
"""
Apply Agent Baseline Results — validates, merges, and saves Agent-generated
test-to-feature mappings as the new tests.json.

Usage:
    python .claude/skills/npu-risk-graph/scripts/apply_agent_baseline.py [--results agent_baseline_results.json]
"""

import sys
from collections import Counter
from pathlib import Path

# Add this script's directory for shared imports (common.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BASELINE_DIR,
    REPO_ROOT,
    ROOT,
    get_current_commit,
    load_json,
    now_iso,
    save_json,
)


# Canonical feature names for validation
def _load_feature_names() -> set[str]:
    data = load_json(BASELINE_DIR / "features.json")
    return {f["name"] for f in data.get("features", [])}


def _discover_test_paths() -> set[str]:
    """All known test file paths."""
    test_root = REPO_ROOT / "test" / "registered" / "ascend"
    if not test_root.exists():
        return set()
    return {
        str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        for p in test_root.rglob("test_*.py")
    }


# ============================================================
# Validation
# ============================================================


def validate_mapping(
    mapping: dict, feature_names: set[str], known_paths: set[str]
) -> list[str]:
    """Validate a single mapping entry. Returns list of warning messages."""
    warnings = []

    # Required fields
    test_file = mapping.get("test_file", "")
    if not test_file:
        return ["MISSING test_file — entry skipped"]

    # Path must exist
    if test_file not in known_paths:
        warnings.append(f"HALLUCINATED PATH: {test_file}")
        return warnings  # hard error, can't recover

    # Features must be valid
    features_tested = mapping.get("features_tested", [])
    if not isinstance(features_tested, list):
        warnings.append(f"features_tested is not a list: {test_file}")
        features_tested = []

    valid_features = []
    for fname in features_tested:
        if fname in feature_names:
            valid_features.append(fname)
        else:
            # Try fuzzy match
            close = _fuzzy_match(fname, feature_names)
            if close:
                warnings.append(
                    f"FEATURE MISMATCH: '{fname}' -> '{close}' (auto-corrected)"
                )
                valid_features.append(close)
            else:
                warnings.append(f"UNKNOWN FEATURE: '{fname}' — dropped")

    mapping["features_tested"] = valid_features

    # Quality score range
    qs = mapping.get("quality_score", 1)
    if not isinstance(qs, (int, float)) or qs < 1 or qs > 5:
        mapping["quality_score"] = max(
            1, min(5, int(qs) if isinstance(qs, (int, float)) else 1)
        )
        warnings.append(
            f"QUALITY CLAMPED: {test_file} qs={qs} -> {mapping['quality_score']}"
        )

    return warnings


def _fuzzy_match(name: str, candidates: set[str]) -> str | None:
    """Find the closest feature name match."""
    name_lower = name.lower().replace("_", "")
    for cand in candidates:
        if cand.lower().replace("_", "") == name_lower:
            return cand
    # Substring match
    for cand in candidates:
        if name_lower in cand.lower().replace("_", ""):
            return cand
        if cand.lower().replace("_", "") in name_lower:
            return cand
    return None


# ============================================================
# Rebuild test records from Agent mappings
# ============================================================


def rebuild_test_records(
    mappings: list[dict], fallback_tests: list[dict] = None
) -> list[dict]:
    """Convert flat Agent mappings into the full tests.json format.

    Uses AST-parsed metadata from the original test file for registration
    info, class names, etc. The Agent provides only features_tested and
    quality_score.
    """
    # Reload original test file metadata via prepare logic
    from prepare_agent_baseline import discover_test_files, extract_test_context

    test_files = discover_test_files()
    ctx_map = {}
    for tf in test_files:
        ctx = extract_test_context(tf)
        ctx_map[ctx["path"]] = ctx

    # Build mapping lookup
    agent_map = {m["test_file"]: m for m in mappings}

    tests = []
    covered = set()
    for path, ctx in sorted(ctx_map.items()):
        am = agent_map.get(path, {})
        if am:
            covered.add(path)

        features_tested = am.get("features_tested", [])
        quality = am.get("quality_score", 1)
        has_oracle = am.get("has_reference_oracle", am.get("has_gsm8k_oracle", False))
        assertion_type = am.get("assertion_type", "threshold")

        # Build test_methods from AST-parsed class info
        test_methods = []
        if ctx.get("test_methods"):
            for m in ctx["test_methods"]:
                test_methods.append(
                    {
                        "name": m,
                        "type": "e2e_correctness" if has_oracle else "e2e_accuracy",
                        "assertion_type": assertion_type,
                        "has_reference_oracle": has_oracle,
                        "model_used": ctx.get("model", ""),
                        "ci_suite": (ctx.get("registrations") or [{}])[0].get(
                            "suite", "unknown"
                        ),
                        "est_time_seconds": (ctx.get("registrations") or [{}])[0].get(
                            "est_time", 0
                        ),
                    }
                )

        tests.append(
            {
                "test_file": path,
                "test_class": (
                    ctx.get("class_names", ["Unknown"])[0]
                    if ctx.get("class_names")
                    else "Unknown"
                ),
                "test_methods": test_methods
                or [
                    {
                        "name": "test_default",
                        "type": "e2e_accuracy",
                        "assertion_type": "threshold",
                        "has_reference_oracle": False,
                        "model_used": "",
                        "ci_suite": "unknown",
                        "est_time_seconds": 0,
                    }
                ],
                "features_tested": features_tested,
                "quality_score": quality,
                "ci_registrations": ctx.get("registrations", []),
                "covers_files": [],
                "coverage_ratio": 0.0,
            }
        )

    # Merge fallback for uncovered files
    if fallback_tests:
        for ft in fallback_tests:
            if ft["test_file"] not in covered:
                tests.append(ft)
                covered.add(ft["test_file"])

    return tests


# ============================================================
# Main
# ============================================================


def apply(results_path: Path = None):
    """Validate Agent results and produce tests.json."""
    if results_path is None:
        results_path = BASELINE_DIR / "agent_baseline_results.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Agent results not found: {results_path}\n" "Run the Workflow first."
        )

    feature_names = _load_feature_names()
    known_paths = _discover_test_paths()

    # Load results
    results_data = load_json(results_path)
    mappings = results_data.get("mappings", [])

    if not mappings:
        raise ValueError("No mappings found in Agent results")

    print(f"Loaded {len(mappings)} mappings from Agent results")

    # Validate
    all_warnings = []
    valid_mappings = []
    for m in mappings:
        warnings = validate_mapping(m, feature_names, known_paths)
        all_warnings.extend(warnings)
        if not any("skipped" in w.lower() for w in warnings):
            valid_mappings.append(m)

    # Report warnings
    if all_warnings:
        print(f"\nValidation warnings ({len(all_warnings)}):")
        for w in all_warnings[:10]:
            print(f"  [WARN] {w}")
        if len(all_warnings) > 10:
            print(f"  ... and {len(all_warnings) - 10} more")

    print(f"\nValid mappings: {len(valid_mappings)}")

    # Rebuild test records. Files not covered by the Agent get empty features_tested.
    missing = known_paths - {m["test_file"] for m in valid_mappings}
    if missing:
        print(
            f"\n{len(missing)} files not covered by Agent (no fallback — will have empty features_tested)"
        )
    tests = rebuild_test_records(valid_mappings)
    print(f"Total test records: {len(tests)}")

    # Quality distribution
    qual_dist = Counter(t.get("quality_score", 0) for t in tests)
    print(f"Quality distribution: {dict(sorted(qual_dist.items()))}")

    # Feature coverage
    feat_count = Counter()
    for t in tests:
        for ft in t.get("features_tested", []):
            feat_count[ft] += 1
    print(f"Features covered: {len(feat_count)}/{len(feature_names)}")
    print(f"Top 5: {feat_count.most_common(5)}")

    # Save
    output = {
        "generated_at": now_iso(),
        "git_commit": get_current_commit(),
        "total_tests": len(tests),
        "tests": tests,
        "_meta": {
            "test_mapping_mode": "workflow_v1",
            "agent_mappings": len(valid_mappings),
            "validation_warnings": len(all_warnings),
        },
    }

    tests_path = BASELINE_DIR / "tests.json"
    save_json(tests_path, output)
    print(f"\nSaved: {tests_path}")

    # Update agent meta
    agent_meta = load_json(BASELINE_DIR / ".agent_meta.json")
    agent_meta["test_mapping_mode"] = "workflow_v1"
    agent_meta["test_mapped_at"] = now_iso()
    save_json(BASELINE_DIR / ".agent_meta.json", agent_meta)

    return output


def _load_features_list() -> list[dict]:
    data = load_json(BASELINE_DIR / "features.json")
    return data.get("features", [])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply Agent baseline results")
    parser.add_argument("--results", default=None, help="Agent results JSON path")
    args = parser.parse_args()

    apply(results_path=Path(args.results) if args.results else None)
