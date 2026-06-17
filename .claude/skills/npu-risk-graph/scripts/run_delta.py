#!/usr/bin/env python3
"""
Incremental Risk Delta Engine

Detects changes since last baseline, computes affected features
via 1-hop JSON + n-hop KG propagation, runs incremental Agent scoring,
and generates a Delta Report.

Usage:
    python .claude/skills/npu-risk-graph/scripts/run_delta.py [--mode pr_check|full_delta]
    pr_check: Fast incremental (~30s), for PR CI gating
    full_delta: Full incremental with KG propagation (~2min)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add this script's directory for shared imports (common.py, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BASELINE_DIR,
    DB_DIR,
    DELTA_DIR,
    GRAPH_DIR,
    REPO_ROOT,
    ROOT,
    RiskLevel,
    affected_features,
    compute_fingerprint,
    factor_score,
    get_changed_files,
    get_current_commit,
    init_directories,
    load_dependency_map,
    load_json,
    now_iso,
    save_json,
)


def load_baseline_profiles() -> dict:
    """Load risk profiles from latest baseline, keyed by feature name."""
    data = load_json(BASELINE_DIR / "risk_profiles.json")
    return {p["feature"]: p for p in data.get("profiles", [])}


def load_baseline_features() -> dict:
    """Load features from latest baseline, keyed by name."""
    data = load_json(BASELINE_DIR / "features.json")
    return {f["name"]: f for f in data.get("features", [])}


def detect_truly_changed(
    feature_set: set[str], features_data: dict, profiles_data: dict
) -> list[str]:
    """Filter features to only those whose source code fingerprint has changed.

    Returns list of feature names that actually need re-scoring.
    """
    truly_changed = []
    for name in feature_set:
        feat = features_data.get(name)
        if not feat:
            truly_changed.append(name)  # New feature, definitely changed
            continue

        old_fp = profiles_data.get(name, {}).get("fingerprint", "")
        new_fp = compute_fingerprint(feat.get("source_files", []))
        if old_fp != new_fp:
            truly_changed.append(name)

    return truly_changed


def load_graph_if_available() -> Optional[object]:
    """Try to load the KG for n-hop propagation. Returns None if unavailable."""
    graph_path = GRAPH_DIR / "latest_graph.json"
    if not graph_path.exists():
        return None
    try:
        import networkx as nx

        data = load_json(graph_path)
        return nx.node_link_graph(data, edges="edges")
    except Exception:
        return None


def kg_propagation(G, directly_affected: set[str], hops: int = 2) -> set[str]:
    """Use KG to find indirectly affected features via DEPENDS_ON /
    SHARES_DEFECT_PATTERN / AFFECTED_BY edges."""
    if G is None:
        return set()

    propagated = set()
    for feat in directly_affected:
        if not G.has_node(feat):
            continue
        # BFS from each directly affected feature
        visited = {feat}
        frontier = {feat}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for _, neighbor, d in G.out_edges(node, data=True):
                    # DEPENDS_ON: explicit dependency coupling
                    if (
                        d.get("type") == "DEPENDS_ON"
                        and G.nodes[neighbor].get("type") == "Feature"
                    ):
                        if (
                            neighbor not in directly_affected
                            and neighbor not in visited
                        ):
                            next_frontier.add(neighbor)
                    # SHARES_DEFECT_PATTERN: systemic risk coupling
                    if d.get("type") == "SHARES_DEFECT_PATTERN":
                        if (
                            neighbor not in directly_affected
                            and neighbor not in visited
                        ):
                            next_frontier.add(neighbor)
                # AFFECTED_BY: shared defect → shared risk
                for u, _, d in G.in_edges(node, data=True):
                    if d.get("type") == "AFFECTED_BY":
                        for _, neighbor, d2 in G.out_edges(u, data=True):
                            if (
                                d2.get("type") == "AFFECTED_BY"
                                and G.nodes[neighbor].get("type") == "Feature"
                            ):
                                if (
                                    neighbor not in directly_affected
                                    and neighbor not in visited
                                ):
                                    next_frontier.add(neighbor)
            visited.update(frontier)
            frontier = next_frontier - visited

        # Catch nodes discovered in the final hop round (they were set as
        # frontier but the loop ended before they could be added to visited).
        visited.update(frontier)
        propagated.update(visited - {feat})

    return propagated


def incremental_rescore(
    features_to_rescore: list[str],
    features_data: dict,
    profiles_data: dict,
    tests_data: dict,
    defect_db: dict,
) -> list[dict]:
    """Re-score only the changed features. Returns list of delta entries."""
    from run_full import (
        compute_risk_profile,  # sys.path already set at module level (line 22)
    )

    deltas = []
    tests_list = tests_data.get("tests", [])

    for name in features_to_rescore:
        old_profile = profiles_data.get(name, {})
        old_score = old_profile.get("composite_score", 0)
        old_level = old_profile.get("level", "unknown")

        feat = features_data.get(name)
        if not feat:
            continue

        new_profile = compute_risk_profile(feat, tests_list, defect_db)
        new_score = new_profile["composite_score"]
        new_level = new_profile["level"]

        diff = new_score - old_score

        # Build delta entry
        delta = {
            "type": "risk_score_change",
            "feature": name,
            "old_score": old_score,
            "new_score": new_score,
            "diff": diff,
            "old_level": old_level,
            "new_level": new_level,
            "direction": (
                "increased" if diff > 0 else ("decreased" if diff < 0 else "unchanged")
            ),
            "probability_change": new_profile["probability"]["total"]
            - old_profile.get("probability", {}).get("total", 0),
            "impact_change": new_profile["impact"]["total"]
            - old_profile.get("impact", {}).get("total", 0),
            "new_profile": new_profile,
        }

        # Determine severity of the change
        if diff >= 8:
            delta["severity"] = "critical_change"
        elif diff >= 4:
            delta["severity"] = "significant_change"
        elif diff >= 1:
            delta["severity"] = "minor_change"
        elif diff <= -5:
            delta["severity"] = "significant_improvement"
        else:
            delta["severity"] = "unchanged"

        deltas.append(delta)

    return deltas


def generate_delta_report(
    deltas: list[dict],
    changed_files: list[str],
    direct_features: set[str],
    propagated_features: set[str],
) -> dict:
    """Generate the complete Delta Record."""
    increased = [d for d in deltas if d["direction"] == "increased"]
    decreased = [d for d in deltas if d["direction"] == "decreased"]
    unchanged = [d for d in deltas if d["direction"] == "unchanged"]

    # Critical increases that need attention
    critical_increases = [
        d
        for d in increased
        if d["new_level"] == "critical" and d["old_level"] != "critical"
    ]

    # Net risk change
    net_change = sum(d["diff"] for d in deltas)

    # Recommendation
    if critical_increases:
        recommendation = (
            "BLOCK — New critical-level risks introduced without adequate testing"
        )
    elif net_change > 5:
        recommendation = (
            "WARNING — Overall risk increased significantly. Consider adding tests."
        )
    elif net_change < -5:
        recommendation = (
            "MERGABLE — Overall risk decreased significantly. Good improvement."
        )
    elif net_change >= 0:
        recommendation = "MERGABLE — Risk is stable or slightly increased."
    else:
        recommendation = "MERGABLE — Overall risk is decreasing."

    return {
        "delta_id": f"d_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "parent_commit": load_json(BASELINE_DIR / "features.json").get(
            "git_commit", "unknown"
        ),
        "current_commit": get_current_commit(),
        "trigger": "manual",  # or "pr_check", "scheduled"
        "changed_files": changed_files,
        "direct_features": sorted(direct_features),
        "propagated_features": sorted(propagated_features),
        "total_features_rescored": len(deltas),
        "changes": deltas,
        "summary": {
            "risk_increased": [d["feature"] for d in increased],
            "risk_decreased": [d["feature"] for d in decreased],
            "risk_unchanged": [d["feature"] for d in unchanged],
            "critical_new": [d["feature"] for d in critical_increases],
            "net_risk_change": net_change,
        },
        "recommendation": recommendation,
        "generated_at": now_iso(),
    }


def generate_pr_comment(report: dict) -> str:
    """Generate a concise PR comment from the delta report."""
    lines = []
    lines.append("## 🤖 NPU Risk Delta Report")
    lines.append("")

    # Summary stats
    s = report["summary"]
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Changed files | {len(report['changed_files'])} |")
    lines.append(f"| Direct features | {len(report['direct_features'])} |")
    lines.append(f"| Propagated features | {len(report['propagated_features'])} |")
    lines.append(f"| Features rescored | {report['total_features_rescored']} |")
    lines.append(f"| Net risk change | {s['net_risk_change']:+d} |")
    lines.append("")

    # Risk changes table
    if report["changes"]:
        lines.append("| Feature | Old | New | Diff | Level Change |")
        lines.append("|---------|-----|-----|------|-------------|")
        for d in report["changes"]:
            emoji = (
                "⬆️"
                if d["direction"] == "increased"
                else ("⬇️" if d["direction"] == "decreased" else "➡️")
            )
            level_change = (
                f"{d['old_level']} → {d['new_level']}"
                if d["old_level"] != d["new_level"]
                else "—"
            )
            lines.append(
                f"| {emoji} {d['feature']} | {d['old_score']} | {d['new_score']} | {d['diff']:+d} | {level_change} |"
            )

    lines.append("")
    lines.append(f"**Assessment**: {report['recommendation']}")

    if s["critical_new"]:
        lines.append(f"\n⚠️ **New critical risks**: {', '.join(s['critical_new'])}")
        lines.append(
            "These features now have critical risk levels. Consider adding tests before merging."
        )

    return "\n".join(lines)


def _try_update_graph(report: dict):
    """Attempt to apply the delta report to the knowledge graph."""
    try:
        from update_graph import apply_delta, load_graph, save_graph, snapshot_graph

        G = load_graph()
        print("[6/6] Updating knowledge graph...")
        snap_path = snapshot_graph(G)
        print(f"  → Snapshot: {snap_path.name}")
        apply_delta(G, report)
        save_graph(G)
        print("  → Graph updated.")
    except FileNotFoundError:
        print(
            "[6/6] Graph not found — skipping graph update. Run build_graph.py first."
        )
    except ImportError as e:
        print(f"[6/6] update_graph module not available — skipping. ({e})")
    except Exception as e:
        print(f"[6/6] Graph update failed: {e}")


# ============================================================
# Main
# ============================================================


def run_delta(mode: str = "pr_check", base_ref: str = "HEAD~1", head_ref: str = "HEAD"):
    """Execute incremental risk analysis."""
    init_directories()

    print(f"\n{'='*60}")
    print(f"NPU Risk Framework — Delta Analysis ({mode})")
    print(f"{'='*60}\n")

    # Step 1: Detect changes
    print("[1/5] Detecting changed files...")
    changed_files = get_changed_files(base_ref, head_ref)
    if not changed_files:
        print("  → No changes detected. Exiting.")
        return None

    npu_related = [
        f
        for f in changed_files
        if any(kw in f for kw in ("npu", "ascend", "NPU", "sglang/srt"))
    ]
    print(f"  → {len(changed_files)} total changed, {len(npu_related)} NPU-related")

    if not npu_related:
        print("  → No NPU-related changes. No risk impact.")
        return None

    # Step 2: Direct feature mapping (1-hop JSON)
    print("[2/5] Mapping files → features (JSON 1-hop)...")
    dep_map = load_dependency_map()
    direct_features = affected_features(npu_related, dep_map)
    print(f"  → Directly affected: {direct_features}")

    # Step 3: KG propagation (if available)
    propagated_features = set()
    if mode == "full_delta":
        print("[3/5] KG n-hop propagation...")
        G = load_graph_if_available()
        if G:
            propagated_features = kg_propagation(G, direct_features)
            print(f"  → Propagated features: {propagated_features}")
        else:
            print("  → KG not available, skipping propagation")
    else:
        print("[3/5] PR check mode — skipping KG propagation (<30s target)")

    # Step 4: Fingerprint filtering
    all_candidate_features = direct_features | propagated_features
    print(
        f"[4/5] Fingerprint filtering {len(all_candidate_features)} candidate features..."
    )
    features_data = load_baseline_features()
    profiles_data = load_baseline_profiles()
    truly_changed = detect_truly_changed(
        all_candidate_features, features_data, profiles_data
    )
    print(f"  → Truly changed (fingerprint differs): {truly_changed}")

    if not truly_changed:
        print("  → No features actually changed. Fingerprints match baseline.")
        # Still produce a delta report (empty changes)
        report = generate_delta_report(
            [], npu_related, direct_features, propagated_features
        )
        report["recommendation"] = (
            "MERGABLE — No risk impact detected (fingerprints unchanged)"
        )
        return report

    # Step 5: Incremental rescore
    print(f"[5/5] Incremental rescore of {len(truly_changed)} features...")
    # Populate change_freq cache before rescore (compute_risk_profile reads from it)
    from run_full import _compute_all_change_freqs

    _compute_all_change_freqs(list(features_data.values()))
    tests_data = load_json(BASELINE_DIR / "tests.json")
    defect_db = load_json(DB_DIR / "defect_db.json")
    deltas = incremental_rescore(
        truly_changed, features_data, profiles_data, tests_data, defect_db
    )

    # Generate report
    report = generate_delta_report(
        deltas, npu_related, direct_features, propagated_features
    )

    # Save
    commit = get_current_commit()
    output_path = DELTA_DIR / f"{commit[:10]}.json"
    save_json(output_path, report)

    # Also save as latest
    latest_path = DELTA_DIR / "latest.json"
    save_json(latest_path, report)

    # Step 6: Update the knowledge graph (if available and requested)
    update_graph_flag = mode in ("full_delta",)  # default: update on full_delta
    if update_graph_flag:
        _try_update_graph(report)

    print(f"\n{'='*60}")
    print(f"Delta Report: {output_path}")
    print(f"  Risk increased: {report['summary']['risk_increased']}")
    print(f"  Risk decreased: {report['summary']['risk_decreased']}")
    print(f"  Net change: {report['summary']['net_risk_change']:+d}")
    print(f"  Recommendation: {report['recommendation']}")
    print(f"{'='*60}\n")

    # Print PR comment
    if mode == "pr_check":
        print("\n--- PR Comment ---")
        print(generate_pr_comment(report))

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPU Risk Delta Analysis")
    parser.add_argument(
        "--mode",
        choices=["pr_check", "full_delta"],
        default="pr_check",
        help="pr_check: fast (~30s); full_delta: with KG propagation (~2min)",
    )
    parser.add_argument(
        "--base-ref", default="origin/main", help="Base git ref for comparison"
    )
    parser.add_argument(
        "--head-ref", default="HEAD", help="Head git ref for comparison"
    )
    args = parser.parse_args()

    run_delta(args.mode, args.base_ref, args.head_ref)
