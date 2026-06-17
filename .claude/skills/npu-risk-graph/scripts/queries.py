#!/usr/bin/env python3
"""
Risk Knowledge Graph Queries (12 standard query patterns).

Usage:
    from queries import blind_spot_detection, perfect_storm, impacted_features
    from build_graph import build_risk_graph

    G = build_risk_graph()
    blinds = blind_spot_detection(G)
"""

import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
from common import GRAPH_DIR, load_json, save_json


def _edges(G, node, data=True):
    """Safe edge iterator for both DiGraph and MultiDiGraph."""
    for e in G.out_edges(node, data=data):
        if len(e) == 3:
            yield e[0], e[1], e[2]
        else:
            yield e[0], e[1], e[2], e[3]


def _in_edges(G, node, data=True):
    """Safe in-edge iterator for both DiGraph and MultiDiGraph."""
    for e in G.in_edges(node, data=data):
        if len(e) == 3:
            yield e[0], e[1], e[2]
        else:
            yield e[0], e[1], e[2], e[3]


def load_graph(graph_path: Path = None) -> nx.MultiDiGraph:
    """Load graph from node-link JSON file, normalized to MultiDiGraph."""
    if graph_path is None:
        graph_path = GRAPH_DIR / "latest_graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph not found: {graph_path}\n"
            "Run '/npu-risk-graph build' first "
            "(python .claude/skills/npu-risk-graph/scripts/build_graph.py)."
        )
    data = load_json(graph_path)
    G = nx.node_link_graph(data, edges="edges")
    # Normalize to MultiDiGraph for consistent edge iteration
    if not isinstance(G, nx.MultiDiGraph):
        G2 = nx.MultiDiGraph()
        G2.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            G2.add_edge(u, v, **d)
        G = G2
    return G


# ============================================================
# Type 1: Cascade Impact Analysis
# ============================================================


def impacted_features(G: nx.MultiDiGraph, file_name: str, hops: int = 2) -> dict:
    """Q1: Given a changed source file, find all impacted features via n-hop propagation.

    Returns:
        { "direct": [feature_names], "propagated_1hop": [...], "propagated_2hop": [...] }
    """
    # Find the SourceFile node
    file_node = None
    for n, d in G.nodes(data=True):
        if d.get("type") == "SourceFile" and file_name in n:
            file_node = n
            break

    if not file_node:
        return {"direct": [], "propagated_1hop": [], "propagated_2hop": []}

    # Direct: Feature ← IMPLEMENTS_IN ← file
    direct = set()
    for u, v, d in G.in_edges(file_node, data=True):
        if d.get("type") == "IMPLEMENTS_IN" and G.nodes[u].get("type") == "Feature":
            direct.add(u)

    # 1-hop: Feature → DEPENDS_ON → Feature
    hop1 = set()
    for feat in direct:
        for _, neighbor, d in G.out_edges(feat, data=True):
            if (
                d.get("type") == "DEPENDS_ON"
                and G.nodes[neighbor].get("type") == "Feature"
            ):
                hop1.add(neighbor)

    # 2-hop
    hop2 = set()
    for feat in hop1:
        for _, neighbor, d in G.out_edges(feat, data=True):
            if (
                d.get("type") == "DEPENDS_ON"
                and G.nodes[neighbor].get("type") == "Feature"
            ):
                if neighbor not in direct:
                    hop2.add(neighbor)

    return {
        "file": file_node,
        "direct": sorted(direct),
        "propagated_1hop": sorted(hop1),
        "propagated_2hop": sorted(hop2),
        "total_impacted": len(direct | hop1 | hop2),
    }


def cann_upgrade_blast_radius(
    G: nx.MultiDiGraph, dep_name: str = "CANN"
) -> list[tuple]:
    """Q2: Which features are affected by a dependency upgrade, sorted by risk?"""
    results = []
    for feat, _, d in G.in_edges(dep_name, data=True):
        if d.get("type") == "DEPENDS_ON" and G.nodes[feat].get("type") == "Feature":
            risk = G.nodes[feat].get("risk_score", 0)
            coupling = d.get("weight", 0)
            results.append((feat, risk, coupling))
    return sorted(results, key=lambda x: x[1], reverse=True)


# ============================================================
# Type 2: Blind Spot Detection
# ============================================================


def blind_spot_detection(
    G: nx.MultiDiGraph, bc_threshold: float = 0.5, risk_threshold: int = 20
) -> list[dict]:
    """Q3: Find high-betweenness, low-risk nodes — structural blind spots.

    These are nodes that sit at critical information-passing positions
    in the graph but have low risk scores — a single failure here could
    cascade widely without being flagged by traditional risk analysis.
    """
    blinds = []
    for node, data in G.nodes(data=True):
        bc = data.get("betweenness", 0.0)
        risk = data.get("risk_score", 0)
        ntype = data.get("type", "unknown")
        if bc > bc_threshold and risk < risk_threshold:
            blinds.append(
                {
                    "node": node,
                    "type": ntype,
                    "betweenness": round(bc, 3),
                    "risk_score": risk,
                    "degree": data.get("degree", 0),
                    "gap": round(bc * 45 - risk, 1),  # expected risk - actual risk
                }
            )

    # For non-Feature nodes that have high betweenness
    non_feature = [b for b in blinds if b["type"] != "Feature"]
    return sorted(blinds, key=lambda x: x["gap"], reverse=True), non_feature


def high_coupling_no_test(G: nx.MultiDiGraph) -> list[dict]:
    """Q4: SourceFiles with many outbound dependencies but zero test coverage."""
    results = []
    for node, data in G.nodes(data=True):
        if data.get("type") != "SourceFile":
            continue
        out_deg = G.out_degree(node)
        has_covers = any(
            d.get("type") == "COVERS" for _, _, d in G.in_edges(node, data=True)
        )
        if out_deg >= 2 and not has_covers:
            results.append(
                {
                    "file": node,
                    "out_degree": out_deg,
                    "risk_score": data.get("risk_score", 0),
                }
            )
    return sorted(results, key=lambda x: x["out_degree"], reverse=True)


# ============================================================
# Type 3: Multi-Dimensional Pattern Matching
# ============================================================


def perfect_storm(G: nx.MultiDiGraph, days: int = 90) -> list[dict]:
    """Q5: Find features matching ALL of:
    - platform_divergence >= 4 (NPU-private)
    - has silent defect history
    - zero effective tests (quality <= 2)
    - recently modified
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    results = []

    for node, data in G.nodes(data=True):
        if data.get("type") != "Feature":
            continue

        # Check platform divergence via IMPLEMENTS_IN edges
        has_divergence = False
        for _, file_node, d in G.out_edges(node, data=True):
            if d.get("type") == "IMPLEMENTS_IN":
                if G.nodes[file_node].get("platform_divergence") == "fully_npu_private":
                    has_divergence = True
                    break

        # Check silent defect history
        has_silent = False
        silent_bugs = []
        for u, _, d in G.in_edges(node, data=True):
            if d.get("type") == "AFFECTED_BY" and G.nodes[u].get("type") == "Defect":
                if G.nodes[u].get("is_silent"):
                    has_silent = True
                    silent_bugs.append(u)

        # Check test effectiveness
        test_edges = [
            d.get("weight", 0)
            for _, _, d in G.out_edges(node, data=True)
            if d.get("type") == "TESTED_BY"
        ]
        max_quality = max(test_edges) if test_edges else 0

        # Check recent modification
        recent = (
            data.get("last_modified", "") > cutoff
            if data.get("last_modified")
            else False
        )

        if has_divergence and (has_silent or max_quality <= 2):
            results.append(
                {
                    "feature": node,
                    "risk_score": data.get("risk_score", 0),
                    "level": data.get("level", "unknown"),
                    "has_silent_defects": has_silent,
                    "silent_bugs": silent_bugs,
                    "best_test_quality": max_quality,
                    "recently_modified": recent,
                    "score": (1 if has_divergence else 0)
                    + (2 if has_silent else 0)
                    + (2 if max_quality <= 2 else 0)
                    + (1 if recent else 0),
                }
            )

    return sorted(results, key=lambda x: x["score"], reverse=True)


def defect_hotspots(G: nx.MultiDiGraph) -> list[dict]:
    """Q6: SourceFiles with the most defect FIXED_IN edges — "repeat offender" modules."""
    file_defects = {}
    for node, data in G.nodes(data=True):
        if data.get("type") != "Defect":
            continue
        for _, file_id, d in G.out_edges(node, data=True):
            if d.get("type") == "FIXED_IN":
                file_defects.setdefault(file_id, []).append(
                    {
                        "bug_id": node,
                        "category": data.get("category", "unknown"),
                        "is_silent": data.get("is_silent", False),
                        "failure_mode": data.get("failure_mode", "unknown"),
                    }
                )

    hotspots = []
    for file_id, bugs in file_defects.items():
        categories = Counter(b["category"] for b in bugs)
        silent_count = sum(1 for b in bugs if b["is_silent"])
        hotspots.append(
            {
                "file": file_id,
                "total_defects": len(bugs),
                "categories": dict(categories),
                "silent_defects": silent_count,
                "silent_ratio": round(silent_count / len(bugs), 2) if bugs else 0,
                "bugs": bugs,
            }
        )

    return sorted(hotspots, key=lambda x: x["total_defects"], reverse=True)


# ============================================================
# Type 4: Test Strategy Optimization
# ============================================================


def test_priority_ranking(G: nx.MultiDiGraph) -> list[dict]:
    """Q7: Rank features by (risk_score × (5 - test_effectiveness)) — where to invest next."""
    priority = []
    for node, data in G.nodes(data=True):
        if data.get("type") != "Feature":
            continue

        test_weights = [
            d.get("weight", 0)
            for _, _, d in G.out_edges(node, data=True)
            if d.get("type") == "TESTED_BY"
        ]
        max_eff = max(test_weights) if test_weights else 0
        gap = 5 - max_eff
        score = data.get("risk_score", 0) * gap

        priority.append(
            {
                "feature": node,
                "risk_score": data.get("risk_score", 0),
                "best_test_quality": max_eff,
                "gap": gap,
                "priority_score": score,
                "test_depth": data.get("test_depth", "L3"),
            }
        )

    return sorted(priority, key=lambda x: x["priority_score"], reverse=True)


# ============================================================
# Type 5: Change Impact Propagation
# ============================================================


def n_hop_impact(G: nx.MultiDiGraph, start_node: str, hops: int = 3) -> dict:
    """Q8: n-hop BFS from a start node, returning all affected nodes by hop distance."""
    visited = set()
    levels = {}
    frontier = {start_node}

    for h in range(hops):
        levels[h] = []
        next_frontier = set()
        for node in frontier:
            for _, neighbor, d in G.out_edges(node, data=True):
                if d.get("type") in (
                    "IMPLEMENTS_IN",
                    "DEPENDS_ON",
                    "AFFECTED_BY",
                    "TESTED_BY",
                    "COVERS",
                    "FIXED_IN",
                ):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        levels[h].append(
                            {
                                "node": neighbor,
                                "type": G.nodes[neighbor].get("type", "unknown"),
                                "edge_type": d.get("type"),
                                "weight": d.get("weight", 0),
                            }
                        )
            # Also check incoming edges
            for u, _, d in G.in_edges(node, data=True):
                if d.get("type") in ("AFFECTED_BY", "WARNS_OF"):
                    if u not in visited:
                        next_frontier.add(u)
                        levels[h].append(
                            {
                                "node": u,
                                "type": G.nodes[u].get("type", "unknown"),
                                "edge_type": d.get("type"),
                                "weight": d.get("weight", 0),
                            }
                        )
        visited.update(frontier)
        frontier = next_frontier - visited

    # Catch nodes discovered in the final hop round (they were set as
    # frontier but the loop ended before they could be added to visited).
    visited.update(frontier)

    return {
        "start_node": start_node,
        "start_type": G.nodes[start_node].get("type", "unknown"),
        "max_hops": hops,
        "total_impacted": len(visited) - 1,
        "levels": {k: v for k, v in levels.items() if v},
    }


# ============================================================
# Type 6: Structural Analysis
# ============================================================


def cross_feature_defect_patterns(G: nx.MultiDiGraph) -> list[dict]:
    """Q9: Failure modes that affect >= 2 Features → systemic risk patterns."""
    pattern_features = {}
    for node, data in G.nodes(data=True):
        if data.get("type") != "Defect":
            continue
        mode = data.get("failure_mode", "unknown")
        pattern_features.setdefault(mode, []).append(node)

    patterns = []
    for mode, bug_ids in pattern_features.items():
        if len(bug_ids) >= 2:
            affected_features = set()
            for bid in bug_ids:
                for _, feat, d in G.out_edges(bid, data=True):
                    if d.get("type") == "AFFECTED_BY":
                        affected_features.add(feat)
            patterns.append(
                {
                    "failure_mode": mode,
                    "defect_count": len(bug_ids),
                    "affected_features": sorted(affected_features),
                    "is_systemic": len(affected_features) >= 3,
                }
            )

    return sorted(patterns, key=lambda x: x["defect_count"], reverse=True)


def test_redundancy_check(G: nx.MultiDiGraph) -> list[dict]:
    """Q10: TestCase pairs that cover the exact same features → potential redundancy."""
    test_features = {}
    for node, data in G.nodes(data=True):
        if data.get("type") != "TestCase":
            continue
        covered = set()
        for _, feat, d in G.out_edges(node, data=True):
            if d.get("type") == "TESTED_BY":
                covered.add(feat)
        if covered:
            test_features[node] = covered

    redundancies = []
    tests = list(test_features.keys())
    for i in range(len(tests)):
        for j in range(i + 1, len(tests)):
            overlap = test_features[tests[i]] & test_features[tests[j]]
            if len(overlap) >= 2:  # Share >= 2 features
                redundancies.append(
                    {
                        "test_a": tests[i],
                        "test_b": tests[j],
                        "shared_features": sorted(overlap),
                        "overlap_count": len(overlap),
                    }
                )

    return sorted(redundancies, key=lambda x: x["overlap_count"], reverse=True)


def longest_dependency_chain(G: nx.MultiDiGraph) -> list[dict]:
    """Q11: Features with the longest DEPENDS_ON chains → most cascade-vulnerable."""
    dep_graph = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if (
            d.get("type") == "DEPENDS_ON"
            and G.nodes[u].get("type") == "Feature"
            and G.nodes[v].get("type") == "Feature"
        ):
            dep_graph.add_edge(u, v)

    chain_lengths = {}
    for node in dep_graph.nodes():
        try:
            lengths = nx.single_source_longest_dag_path_length(dep_graph, node)
            chain_lengths[node] = max(lengths.values()) if lengths else 0
        except Exception:
            chain_lengths[node] = 0

    results = []
    for node, length in chain_lengths.items():
        if G.has_node(node):
            results.append(
                {
                    "feature": node,
                    "chain_length": length,
                    "risk_score": G.nodes[node].get("risk_score", 0),
                    "composite_vulnerability": length
                    * G.nodes[node].get("risk_score", 0),
                }
            )

    return sorted(results, key=lambda x: x["composite_vulnerability"], reverse=True)


def orphan_tests(G: nx.MultiDiGraph) -> list[str]:
    """Q12: TestCase nodes with no TESTED_BY edges → misconfigured tests."""
    orphans = []
    for node, data in G.nodes(data=True):
        if data.get("type") != "TestCase":
            continue
        in_edges = list(G.in_edges(node))
        out_feature_edges = [
            (u, v)
            for u, v, d in G.out_edges(node, data=True)
            if d.get("type") == "TESTED_BY"
        ]
        if not in_edges and not out_feature_edges:
            orphans.append(node)

    # Also check: TestCase has no TESTED_BY edge to any Feature
    for node, data in G.nodes(data=True):
        if data.get("type") != "TestCase":
            continue
        has_feature_edge = False
        for _, feat, d in G.out_edges(node, data=True):
            if d.get("type") == "TESTED_BY" and G.nodes[feat].get("type") == "Feature":
                has_feature_edge = True
                break
        if not has_feature_edge and node not in orphans:
            orphans.append(node)

    return orphans


# ============================================================
# Single-Feature Risk Report
# ============================================================

# ============================================================
# Testing-Knowledge Queries (OraclePattern / FailureMode)
# ============================================================


def oracle_blind_spots(G: nx.MultiDiGraph, test_id: str = None) -> list[dict]:
    """Q13: What failure modes is a test (or all tests) blind to?

    For each test or a specific test, follows TEST_USES_PATTERN edges
    to find OraclePattern nodes, then PATTERN_BLIND_SPOT edges to find
    FailureMode nodes that the test's oracle patterns cannot detect.

    Args:
        G: The risk graph.
        test_id: Optional test node ID. If None, returns blind spots
                 aggregated across all TestCase nodes.

    Returns:
        List of dicts with test_id, pattern, blind_spot failure mode,
        reason, and suggested mitigation pattern.
    """
    results = []
    test_ids = (
        [test_id]
        if test_id
        else [n for n, d in G.nodes(data=True) if d.get("type") == "TestCase"]
    )

    for tid in test_ids:
        if not G.has_node(tid):
            continue
        # Which oracle patterns does this test use?
        for _, pattern_id, d in G.out_edges(tid, data=True):
            if d.get("type") != "TEST_USES_PATTERN":
                continue
            # What is this pattern blind to?
            for _, fm_id, bd in G.out_edges(pattern_id, data=True):
                if bd.get("type") != "PATTERN_BLIND_SPOT":
                    continue
                fm_node = G.nodes.get(fm_id, {})
                results.append(
                    {
                        "test_id": tid,
                        "test_class": G.nodes[tid].get("test_class", ""),
                        "oracle_pattern": pattern_id,
                        "pattern_name": G.nodes[pattern_id].get("name", pattern_id),
                        "blind_spot": fm_id,
                        "blind_spot_name": fm_node.get("name", fm_id),
                        "is_silent": fm_node.get("is_silent", False),
                        "reason": bd.get("reason", ""),
                        "mitigation": bd.get("mitigation", ""),
                    }
                )

    return sorted(results, key=lambda x: x["is_silent"], reverse=True)


def recommend_oracle_for_feature(G: nx.MultiDiGraph, feature_name: str) -> dict:
    """Q14: Recommend oracle patterns for a feature based on its defect profile.

    Analyzes the feature's historical defects to determine which failure
    modes are present, then recommends oracle patterns that can detect
    those modes (and warns about patterns that would miss them).

    Returns a dict with:
      - ``required_patterns``: oracle patterns needed to cover all defect modes
      - ``current_patterns``: patterns already used by existing tests
      - ``coverage_gap``: failure modes with no oracle pattern coverage
      - ``recommendation``: human-readable next step
    """
    if not G.has_node(feature_name):
        return {"error": f"Feature '{feature_name}' not found"}

    # ── Collect failure modes from this feature's defects ──
    defect_modes = set()
    has_silent = False
    for u, v, k, d in G.in_edges(feature_name, keys=True, data=True):
        if d.get("type") == "AFFECTED_BY":
            dn = G.nodes[u]
            mode = dn.get("failure_mode", "unknown")
            defect_modes.add(mode)
            if dn.get("is_silent"):
                has_silent = True

    # ── Build reverse index: FailureMode → OraclePatterns that detect it ──
    fm_to_patterns: dict[str, list[dict]] = {}
    for _, pattern_id, d in G.out_edges(data=True):
        if d.get("type") == "PATTERN_DETECTS":
            fm_to_patterns.setdefault(pattern_id, []).append(
                {
                    "pattern_id": d.get("source"),  # FIXME: edge direction
                }
            )

    # Walk PATTERN_DETECTS edges (OraclePattern → FailureMode)
    # Skip patterns not supported on NPU.
    fm_to_patterns = {}
    for node, data in G.nodes(data=True):
        if data.get("type") != "OraclePattern":
            continue
        if data.get("npu_supported") is False:
            continue
        for _, fm_id, d in G.out_edges(node, data=True):
            if d.get("type") == "PATTERN_DETECTS":
                fm_to_patterns.setdefault(fm_id, []).append(
                    {
                        "pattern_id": node,
                        "pattern_name": data.get("name", node),
                        "confidence": d.get("confidence", 1.0),
                        "severity": data.get("severity", "unknown"),
                    }
                )

    # ── Also: which tests already cover this feature? ──
    current_patterns = set()
    current_tests = []
    for _, test_id, d in G.out_edges(feature_name, data=True):
        if d.get("type") != "TESTED_BY":
            continue
        tn = G.nodes[test_id]
        current_tests.append(
            {
                "test_id": test_id,
                "test_class": tn.get("test_class", ""),
                "quality_score": tn.get("quality_score", 0),
                "has_oracle": tn.get("has_oracle", False),
            }
        )
        for _, pattern_id, pd in G.out_edges(test_id, data=True):
            if pd.get("type") == "TEST_USES_PATTERN":
                current_patterns.add(pattern_id)

    # ── Recommend patterns for each detected failure mode ──
    required_patterns = {}
    for mode in defect_modes:
        candidates = fm_to_patterns.get(mode, [])
        required_patterns[mode] = candidates

    # ── Coverage gap: modes with no pattern that can detect them ──
    coverage_gap = []
    for mode in defect_modes:
        if mode not in fm_to_patterns:
            coverage_gap.append(mode)

    # ── Best-guess recommendation ──
    all_recommended = set()
    for mode, patterns in required_patterns.items():
        for p in patterns:
            all_recommended.add(p["pattern_id"])

    missing = all_recommended - current_patterns

    if not defect_modes:
        recommendation = (
            "No defects recorded — start with `exact_match_short` "
            "for algorithmic correctness verification."
        )
    elif has_silent and "logprob_rescore" in missing:
        recommendation = (
            "Feature has silent precision-loss defects. "
            "Add `logprob_rescore` oracle — exact match alone cannot "
            "detect these (evidence: eagle3 investigation, BUG-2026-098)."
        )
    elif missing:
        rec_names = ", ".join(sorted(missing)[:3])
        recommendation = (
            f"Add oracle patterns: {rec_names} to cover "
            f"failure modes: {', '.join(sorted(defect_modes))}"
        )
    else:
        recommendation = (
            "All defect failure modes are covered by existing oracle patterns."
        )

    return {
        "feature": feature_name,
        "defect_failure_modes": sorted(defect_modes),
        "has_silent_defects": has_silent,
        "required_patterns": {
            mode: [p["pattern_name"] for p in patterns]
            for mode, patterns in required_patterns.items()
        },
        "current_tests": current_tests,
        "current_patterns": sorted(current_patterns),
        "missing_patterns": sorted(missing),
        "coverage_gap": coverage_gap,
        "recommendation": recommendation,
    }


def test_pattern_coverage(G: nx.MultiDiGraph) -> dict:
    """Q15: Overview of oracle pattern usage across all tests.

    Returns:
        Dict with per-pattern usage stats, per-feature coverage matrix,
        and a list of features with zero oracle pattern coverage.
    """
    # ── Per-pattern stats ──
    pattern_usage = {}
    for node, data in G.nodes(data=True):
        if data.get("type") != "OraclePattern":
            continue
        tests_using = []
        for _, test_id, d in G.in_edges(node, data=True):
            if d.get("type") == "TEST_USES_PATTERN":
                tests_using.append(
                    {
                        "test_id": test_id,
                        "role": d.get("role", "unknown"),
                        "weight": d.get("weight", 1.0),
                    }
                )

        # What failure modes does this pattern detect / miss?
        detects = []
        blind_to = []
        for _, fm_id, d in G.out_edges(node, data=True):
            if d.get("type") == "PATTERN_DETECTS":
                detects.append(fm_id)
            elif d.get("type") == "PATTERN_BLIND_SPOT":
                blind_to.append(
                    {
                        "mode": fm_id,
                        "reason": d.get("reason", ""),
                        "mitigation": d.get("mitigation", ""),
                    }
                )

        pattern_usage[node] = {
            "name": data.get("name", node),
            "category": data.get("category", "unknown"),
            "severity": data.get("severity", "unknown"),
            "npu_validated": data.get("npu_validated", False),
            "tests_using": tests_using,
            "n_tests": len(tests_using),
            "detects": detects,
            "blind_to": blind_to,
        }

    # ── Features with zero oracle pattern coverage ──
    uncovered_features = []
    for node, data in G.nodes(data=True):
        if data.get("type") != "Feature":
            continue
        tests = [
            v
            for _, v, d in G.out_edges(node, data=True)
            if d.get("type") == "TESTED_BY"
        ]
        has_oracle_pattern = False
        for test_id in tests:
            for _, pattern_id, d in G.out_edges(test_id, data=True):
                if d.get("type") == "TEST_USES_PATTERN":
                    has_oracle_pattern = True
                    break
            if has_oracle_pattern:
                break
        if tests and not has_oracle_pattern:
            uncovered_features.append(
                {
                    "feature": node,
                    "risk_score": data.get("risk_score", 0),
                    "n_tests": len(tests),
                }
            )

    return {
        "pattern_usage": pattern_usage,
        "n_patterns": len(pattern_usage),
        "uncovered_features": sorted(
            uncovered_features, key=lambda x: x["risk_score"], reverse=True
        ),
        "n_uncovered_features": len(uncovered_features),
    }


def feature_risk_report(
    G: nx.MultiDiGraph,
    feature_name: str,
    profiles_data: dict = None,
) -> dict:
    """Generate a comprehensive risk report for a single feature.

    Combines graph traversal (defects, tests, dependencies, shared patterns,
    source files) with risk factor details from risk_profiles.json.

    Returns a structured dict; use ``format_feature_report()`` to render as
    Markdown or pass ``--json`` on the CLI.
    """
    if not G.has_node(feature_name):
        return {
            "error": f"Feature '{feature_name}' not found in graph",
            "available_features": sorted(
                n for n, d in G.nodes(data=True) if d.get("type") == "Feature"
            )[:20],
        }

    node = G.nodes[feature_name]
    if node.get("type") != "Feature":
        return {
            "error": f"Node '{feature_name}' is a {node.get('type')}, not a Feature"
        }

    # --- Profile data from risk_profiles.json ---
    profiles = profiles_data or {}
    profile = profiles.get(feature_name, {})
    prob_factors = profile.get("probability", {}).get("factors", {})
    imp_factors = profile.get("impact", {}).get("factors", {})

    # --- Defects (AFFECTED_BY: Defect -> Feature) ---
    defects = []
    has_silent = False
    for u, v, k, d in G.in_edges(feature_name, keys=True, data=True):
        if d.get("type") == "AFFECTED_BY" and G.nodes[u].get("type") == "Defect":
            dn = G.nodes[u]
            defects.append(
                {
                    "bug_id": u,
                    "title": dn.get("title", ""),
                    "root_cause": dn.get("root_cause", ""),
                    "category": dn.get("category", "unknown"),
                    "severity": dn.get("severity", 3),
                    "is_silent": dn.get("is_silent", False),
                    "failure_mode": dn.get("failure_mode", "unknown"),
                    "date_fixed": dn.get("date_fixed", ""),
                    "confidence": dn.get("confidence", 0.5),
                }
            )
            if dn.get("is_silent"):
                has_silent = True
    defects.sort(key=lambda x: x.get("date_fixed", ""), reverse=True)

    # --- Tests (TESTED_BY: Feature -> TestCase) ---
    tests = []
    for u, v, k, d in G.out_edges(feature_name, keys=True, data=True):
        if d.get("type") == "TESTED_BY" and G.nodes[v].get("type") == "TestCase":
            tn = G.nodes[v]
            tests.append(
                {
                    "test_file": v,
                    "test_class": tn.get("test_class", ""),
                    "test_type": tn.get("test_type", "unknown"),
                    "quality_score": tn.get("quality_score", 0),
                    "has_oracle": tn.get("has_oracle", False),
                    "assertion_type": tn.get("assertion_type", "unknown"),
                    "n_methods": tn.get("n_methods", 0),
                    "ci_suite": tn.get("ci_suite", "unknown"),
                    "est_time_seconds": tn.get("est_time_seconds", 0),
                }
            )
    tests.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # --- Dependencies (DEPENDS_ON: Feature -> Feature/Dependency) ---
    depends_on = []
    depended_by = []
    for u, v, k, d in G.out_edges(feature_name, keys=True, data=True):
        if d.get("type") == "DEPENDS_ON":
            target_type = G.nodes[v].get("type", "?")
            depends_on.append(
                {
                    "name": v,
                    "type": target_type,
                    "weight": d.get("weight", 0),
                    "via": d.get("via", "unknown"),
                }
            )
    for u, v, k, d in G.in_edges(feature_name, keys=True, data=True):
        if d.get("type") == "DEPENDS_ON" and G.nodes[u].get("type") == "Feature":
            depended_by.append(
                {
                    "name": u,
                    "weight": d.get("weight", 0),
                    "via": d.get("via", "unknown"),
                }
            )
    depends_on.sort(key=lambda x: x["weight"], reverse=True)
    depended_by.sort(key=lambda x: x["weight"], reverse=True)

    # --- Shared Defect Patterns (SHARES_DEFECT_PATTERN: Feature <-> Feature) ---
    shared_patterns = []
    for u, v, k, d in G.out_edges(feature_name, keys=True, data=True):
        if d.get("type") == "SHARES_DEFECT_PATTERN":
            shared_patterns.append(
                {
                    "feature": v,
                    "shared_defect_count": d.get("shared_defect_count", 0),
                    "shared_failure_modes": d.get("shared_failure_modes", []),
                    "weight": d.get("weight", 0),
                }
            )
    shared_patterns.sort(key=lambda x: x["shared_defect_count"], reverse=True)

    # --- Source Files (IMPLEMENTS_IN: Feature -> SourceFile) ---
    source_files = []
    for u, v, k, d in G.out_edges(feature_name, keys=True, data=True):
        if d.get("type") == "IMPLEMENTS_IN" and G.nodes[v].get("type") == "SourceFile":
            sn = G.nodes[v]
            source_files.append(
                {
                    "path": v,
                    "file_name": sn.get("file_name", ""),
                    "file_type": sn.get("file_type", "unknown"),
                    "lines_of_code": sn.get("lines_of_code", 0),
                    "n_bugs": sn.get("n_bugs", 0),
                    "platform_divergence": sn.get("platform_divergence", ""),
                }
            )
    source_files.sort(key=lambda x: x["n_bugs"], reverse=True)

    # --- Recommendations ---
    recommendations = _generate_recommendations(
        G.nodes[feature_name],
        defects,
        tests,
        depended_by,
    )

    return {
        "feature": feature_name,
        "category": node.get("category", "unknown"),
        "risk_score": node.get("risk_score", 0),
        "probability": node.get("probability", 0),
        "impact": node.get("impact", 0),
        "level": node.get("level", "unknown"),
        "test_depth": node.get("test_depth", "L3"),
        "complexity": node.get("complexity", 1),
        "n_tests": node.get("n_tests", 0),
        "n_effective_tests": node.get("n_effective_tests", 0),
        "n_source_files": node.get("n_source_files", 0),
        "betweenness": node.get("betweenness", 0.0),
        "degree": node.get("degree", 0),
        "last_modified": node.get("last_modified", ""),
        "risk_factors": {
            "probability": {
                "total": profile.get("probability", {}).get("total", 0),
                "max": profile.get("probability", {}).get("max", 25),
                "factors": prob_factors,
            },
            "impact": {
                "total": profile.get("impact", {}).get("total", 0),
                "max": profile.get("impact", {}).get("max", 20),
                "factors": imp_factors,
            },
        },
        "defects": defects,
        "defect_count": len(defects),
        "has_silent_defects": has_silent,
        "tests": tests,
        "test_count": len(tests),
        "best_test_quality": max((t.get("quality_score", 0) for t in tests), default=0),
        "avg_test_quality": round(
            sum(t.get("quality_score", 0) for t in tests) / max(1, len(tests)), 1
        ),
        "depends_on": depends_on,
        "depended_by": depended_by,
        "shared_defect_patterns": shared_patterns,
        "source_files": source_files,
        "recommendations": recommendations,
    }


def _generate_recommendations(
    node: dict,
    defects: list[dict],
    tests: list[dict],
    depended_by: list[dict],
) -> list[dict]:
    """Generate rule-based recommendations for a feature."""
    recs = []
    risk_score = node.get("risk_score", 0)
    n_effective = node.get("n_effective_tests", 0)
    n_tests = node.get("n_tests", 0)
    n_defects = len(defects)
    has_silent = any(d.get("is_silent") for d in defects)
    betweenness = node.get("betweenness", 0.0)
    max_quality = max((t.get("quality_score", 0) for t in tests), default=0)

    # Urgency levels
    if risk_score >= 28 and n_effective == 0:
        recs.append(
            {
                "level": "critical",
                "icon": "🔴",
                "message": f"Critical risk (score={risk_score}) with 0 effective tests — write tests urgently",
            }
        )
    elif risk_score >= 20 and n_effective == 0:
        recs.append(
            {
                "level": "high",
                "icon": "🟠",
                "message": f"High risk (score={risk_score}) with 0 effective tests — prioritize test coverage",
            }
        )

    if has_silent and n_tests == 0:
        recs.append(
            {
                "level": "high",
                "icon": "⚠️",
                "message": f"{sum(1 for d in defects if d.get('is_silent'))} silent defects and no tests — add output precision verification",
            }
        )
    elif has_silent:
        recs.append(
            {
                "level": "medium",
                "icon": "⚠️",
                "message": f"{sum(1 for d in defects if d.get('is_silent'))} silent defects — ensure tests have reference oracle",
            }
        )

    if n_defects >= 5:
        recs.append(
            {
                "level": "high",
                "icon": "🟠",
                "message": f"{n_defects} historical defects — consider targeted refactoring or expanded regression suite",
            }
        )
    elif n_defects >= 3:
        recs.append(
            {
                "level": "medium",
                "icon": "🟡",
                "message": f"{n_defects} historical defects — track defect trend over time",
            }
        )

    if betweenness > 0.05 and n_tests == 0:
        recs.append(
            {
                "level": "high",
                "icon": "🔗",
                "message": f"High betweenness ({betweenness:.3f}) with no tests — failure cascades to {len(depended_by)} dependent features",
            }
        )
    elif betweenness > 0.03:
        recs.append(
            {
                "level": "medium",
                "icon": "🔗",
                "message": f"Elevated betweenness ({betweenness:.3f}) — prioritize in regression test selection",
            }
        )

    if n_tests > 0 and max_quality < 3:
        recs.append(
            {
                "level": "medium",
                "icon": "📉",
                "message": f"Best test quality is {max_quality}/5 — tests may be superficial (accuracy checks only)",
            }
        )

    if n_tests == 0 and n_defects == 0:
        recs.append(
            {
                "level": "info",
                "icon": "ℹ️",
                "message": "No defects recorded and no tests — monitor for first defect signal",
            }
        )

    return recs


def prepare_agent_bundle(
    G: nx.MultiDiGraph,
    keyword: str,
    profiles_data: dict = None,
) -> dict:
    """Prepare a data bundle for the Agent to semantically classify sub-feature items.

    Finds the best-matching parent feature for *keyword*, exports all its
    source_files / defects / tests together with profile data, and returns a
    structured dict (intended to be saved as JSON and fed to an Agent with a
    classification schema).

    Also saves the bundle to ``.sglang-risk/graph/agent_bundle_{keyword}.json``
    (data directory) so the Agent path can be resumed.

    Returns a dict with:
      - ``bundle``: the classification payload
      - ``bundle_path``: absolute filesystem path to the saved JSON
      - ``workflow_instruction``: human-readable next-step message
      - ``error``: if no parent feature matched
    """
    keyword_lower = keyword.lower()
    # Normalize spaces ↔ underscores so "speculative decoding" matches
    # "speculative_decoding" (the graph uses underscores for all identifiers).
    keyword_folded = keyword_lower.replace(" ", "_")

    def _kw_in(text: str) -> bool:
        """Return True if the keyword (in either form) appears in *text*."""
        t = text.lower()
        return keyword_lower in t or keyword_folded in t

    # --- Find candidate parent features ---
    candidates = []  # (name, score, reasons)
    for node, data in G.nodes(data=True):
        if data.get("type") != "Feature":
            continue
        reasons = []
        score = 0
        node_name = str(node)

        # Check feature name
        if _kw_in(node_name):
            reasons.append("name")
            score += 30

        # Check source file paths
        sf_matches = 0
        for u, v, k, d in G.out_edges(node, keys=True, data=True):
            if d.get("type") == "IMPLEMENTS_IN":
                if _kw_in(str(v)):
                    sf_matches += 1
        if sf_matches > 0:
            reasons.append(f"source_files({sf_matches})")
            score += min(sf_matches * 5, 25)

        # Check description
        if _kw_in(data.get("description", "")):
            reasons.append("description")
            score += 10

        if score > 0:
            candidates.append((node_name, score, reasons))

    # --- Fallback: stem matching to find parent feature ---
    # Exact keyword matching above is preferred.  Only when it produces zero
    # candidates do we strip trailing digits (e.g. "eagle3" → "eagle") as a
    # best-effort heuristic to locate a parent feature.  The stem is ONLY used
    # for parent discovery — the Agent does the actual semantic scoping later.
    if not candidates:
        import re as _re

        keyword_stem = _re.sub(r"\d+$", "", keyword_lower).rstrip("_-")
        if len(keyword_stem) >= 3 and keyword_stem != keyword_lower:
            for node, data in G.nodes(data=True):
                if data.get("type") != "Feature":
                    continue
                score = 0
                reasons = []
                if keyword_stem in str(node).lower():
                    reasons.append("name_stem")
                    score += 15
                sf_matches = 0
                for _, v, _k2, d in G.out_edges(node, keys=True, data=True):
                    if d.get("type") == "IMPLEMENTS_IN":
                        if keyword_stem in str(v).lower():
                            sf_matches += 1
                if sf_matches > 0:
                    reasons.append(f"source_files_stem({sf_matches})")
                    score += min(sf_matches * 5, 25)
                if keyword_stem in data.get("description", "").lower():
                    reasons.append("description_stem")
                    score += 10
                if score > 0:
                    candidates.append((str(node), score, reasons))

    if not candidates:
        return {
            "error": f"No feature found matching keyword '{keyword}'.",
            "hint": "Use a more specific keyword, list features with --report, or try a different query.",
        }

    candidates.sort(key=lambda x: x[1], reverse=True)
    parent_name = candidates[0][0]

    # --- Extract full data from parent ---
    node = G.nodes[parent_name]
    profiles = profiles_data or {}
    profile = profiles.get(parent_name, {})

    source_files = []
    for u, v, k, d in G.out_edges(parent_name, keys=True, data=True):
        if d.get("type") == "IMPLEMENTS_IN":
            sn = G.nodes[v]
            source_files.append(
                {
                    "path": str(v),
                    "lines": sn.get("lines_of_code", 0),
                    "n_bugs": sn.get("n_bugs", 0),
                    "platform": sn.get("platform_divergence", ""),
                }
            )

    defects = []
    for u, v, k, d in G.in_edges(parent_name, keys=True, data=True):
        if d.get("type") == "AFFECTED_BY":
            dn = G.nodes[u]
            defects.append(
                {
                    "bug_id": str(u),
                    "title": dn.get("title", ""),
                    "root_cause": dn.get("root_cause", ""),
                    "pr_title": dn.get("pr_title", ""),
                    "pr_body": (dn.get("pr_body", "") or "")[:800],
                    "category": dn.get("category", ""),
                    "severity": dn.get("severity", 3),
                    "is_silent": dn.get("is_silent", False),
                    "date_fixed": dn.get("date_fixed", ""),
                }
            )

    tests = []
    for u, v, k, d in G.out_edges(parent_name, keys=True, data=True):
        if d.get("type") == "TESTED_BY":
            tn = G.nodes[v]
            tests.append(
                {
                    "path": str(v),
                    "test_class": tn.get("test_class", ""),
                    "test_type": tn.get("test_type", ""),
                    "quality": tn.get("quality_score", 0),
                    "has_oracle": tn.get("has_oracle", False),
                }
            )

    bundle = {
        "parent_feature": parent_name,
        "parent_description": node.get("description", ""),
        "parent_complexity": node.get("complexity", 1),
        "parent_risk_score": node.get("risk_score", 0),
        "profile": profile,
        "source_files": source_files,
        "defects": defects,
        "tests": tests,
    }

    # Save to cache file
    bundle_path = GRAPH_DIR / f"agent_bundle_{keyword_lower}.json"
    save_json(bundle_path, bundle)

    workflow_instruction = (
        f"Agent bundle prepared for '{keyword}' (parent: {parent_name}).\n"
        f"Bundle saved to: {bundle_path}\n"
        f"Next: run Agent with classification schema, then:\n"
        f"  python .claude/skills/npu-risk-graph/scripts/queries.py --apply-agent {bundle_path} --keyword {keyword}"
    )

    return {
        "bundle": bundle,
        "bundle_path": str(bundle_path),
        "parent_feature": parent_name,
        "keyword": keyword,
        "workflow_instruction": workflow_instruction,
    }


def apply_agent_result(
    G: nx.MultiDiGraph,
    result_path: str,
    keyword: str = "",
    profiles_data: dict = None,
) -> dict:
    """Apply an Agent's semantic classification result to produce a scoped feature report.

    Reads the Agent's classification JSON (produced from a bundle created by
    ``prepare_agent_bundle``), filters the parent feature's data to only items the
    Agent marked as scoped, recalculates risk scores, and returns a report dict
    compatible with ``format_feature_report()``.

    Args:
        G: The risk graph.
        result_path: Path to the Agent classification JSON file.
        keyword: The original sub-feature keyword (for display).
        profiles_data: Optional risk profiles dict.

    Returns:
        A report dict with the same shape as ``feature_risk_report``, plus a
        ``scoping`` key with parent / keyword / filter metadata.
    """
    classification = load_json(Path(result_path))
    profiles = profiles_data or {}

    # Load the bundle to recover parent metadata
    bundle_path = Path(result_path)
    bundle = load_json(bundle_path) if bundle_path.exists() else {}
    # If result_path is an agent classification result (not a bundle), derive parent
    parent_name = bundle.get("parent_feature", "")
    if not parent_name:
        # Try to find the bundle in GRAPH_DIR (where prepare_agent_bundle saves it)
        for p in list(bundle_path.parent.glob("agent_bundle_*.json")) + list(
            GRAPH_DIR.glob("agent_bundle_*.json")
        ):
            candidate = load_json(p)
            if candidate.get("parent_feature"):
                parent_name = candidate["parent_feature"]
                bundle = candidate
                break
    if not parent_name:
        # Last resort: search by keyword in feature names
        for node, data in G.nodes(data=True):
            if data.get("type") == "Feature":
                for u, v, k, d in G.out_edges(node, keys=True, data=True):
                    if (
                        d.get("type") == "IMPLEMENTS_IN"
                        and keyword.lower() in str(v).lower()
                    ):
                        parent_name = str(node)
                        break
                if parent_name:
                    break
    if not parent_name:
        return {
            "error": "Could not determine parent feature from classification result."
        }

    profile = profiles.get(parent_name, {})
    parent_data = G.nodes[parent_name]

    # Collect scoped items from Agent classification
    scoped_sources = []
    all_sources_count = 0
    if "source_files" in classification:
        sf_items = classification["source_files"]
        all_sources_count = len(sf_items)
        for s in sf_items:
            if s.get("is_scoped", False):
                # Look up full node data
                sf_path = s["path"]
                if G.has_node(sf_path):
                    sn = G.nodes[sf_path]
                    scoped_sources.append(
                        {
                            "path": sf_path,
                            "file_name": sn.get("file_name", ""),
                            "file_type": sn.get("file_type", "unknown"),
                            "lines_of_code": sn.get("lines_of_code", 0),
                            "n_bugs": sn.get("n_bugs", 0),
                            "platform_divergence": sn.get("platform_divergence", ""),
                        }
                    )

    scoped_defects = []
    all_defects_count = 0
    if "defects" in classification:
        df_items = classification["defects"]
        all_defects_count = len(df_items)
        for d in df_items:
            if d.get("is_scoped", False):
                bug_id = d["bug_id"]
                if G.has_node(bug_id):
                    dn = G.nodes[bug_id]
                    scoped_defects.append(
                        {
                            "bug_id": bug_id,
                            "title": dn.get("title", ""),
                            "root_cause": dn.get("root_cause", ""),
                            "category": dn.get("category", "unknown"),
                            "severity": dn.get("severity", 3),
                            "is_silent": dn.get("is_silent", False),
                            "failure_mode": dn.get("failure_mode", "unknown"),
                            "date_fixed": dn.get("date_fixed", ""),
                            "confidence": dn.get("confidence", 0.5),
                        }
                    )

    scoped_tests = []
    all_tests_count = 0
    if "tests" in classification:
        t_items = classification["tests"]
        all_tests_count = len(t_items)
        for t in t_items:
            if t.get("is_scoped", False):
                test_path = t["path"]
                if G.has_node(test_path):
                    tn = G.nodes[test_path]
                    scoped_tests.append(
                        {
                            "test_file": test_path,
                            "test_class": tn.get("test_class", ""),
                            "test_type": tn.get("test_type", "unknown"),
                            "quality_score": tn.get("quality_score", 0),
                            "has_oracle": tn.get("has_oracle", False),
                            "assertion_type": tn.get("assertion_type", "unknown"),
                            "n_methods": tn.get("n_methods", 0),
                            "ci_suite": tn.get("ci_suite", "unknown"),
                            "est_time_seconds": tn.get("est_time_seconds", 0),
                        }
                    )

    scoped_defects.sort(key=lambda x: x.get("date_fixed", ""), reverse=True)
    scoped_tests.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    scoped_sources.sort(key=lambda x: x["n_bugs"], reverse=True)

    has_silent = any(d.get("is_silent") for d in scoped_defects)

    # --- Recalculate risk scores ---
    complexity = parent_data.get("complexity", 1)
    prob_factors = profile.get("probability", {}).get("factors", {})
    imp_factors = profile.get("impact", {}).get("factors", {})

    code_complexity = complexity
    change_freq = (
        prob_factors.get("change_frequency", {}).get("score", 2)
        if isinstance(prob_factors.get("change_frequency"), dict)
        else 2
    )
    dep_depth = (
        prob_factors.get("dependency_depth", {}).get("score", 1)
        if isinstance(prob_factors.get("dependency_depth"), dict)
        else 1
    )

    # Defect score: proportional to scoped/parent ratio
    parent_defects_total = (
        prob_factors.get("historical_defects", {})
        .get("details", {})
        .get("total_bugs", all_defects_count)
    )
    if isinstance(prob_factors.get("historical_defects"), dict):
        parent_defect_score = prob_factors["historical_defects"].get("score", 3)
    else:
        parent_defect_score = 3
    if parent_defects_total == 0:
        defect_score = 0
    elif len(scoped_defects) == 0:
        defect_score = 0
    else:
        defect_score = max(
            1, round(parent_defect_score * len(scoped_defects) / parent_defects_total)
        )

    # Test score: based on scoped test quality
    best_quality = max((t.get("quality_score", 0) for t in scoped_tests), default=0)
    if not scoped_tests:
        test_score = 1
    elif best_quality >= 4:
        test_score = 2 if any(not t.get("has_oracle") for t in scoped_tests) else 3
    elif best_quality >= 3:
        test_score = 3
    else:
        test_score = 4

    raw_prob = code_complexity + change_freq + defect_score + dep_depth + test_score
    probability = min(raw_prob, 25)

    user_exp = (
        imp_factors.get("user_exposure", {}).get("score", 3)
        if isinstance(imp_factors.get("user_exposure"), dict)
        else 3
    )
    fail_mode = (
        imp_factors.get("failure_mode", {}).get("score", 3)
        if isinstance(imp_factors.get("failure_mode"), dict)
        else 3
    )
    impact = min(user_exp + fail_mode, 20)
    risk_score = probability + impact

    if risk_score >= 28:
        level = "critical"
        test_depth = "L0"
    elif risk_score >= 20:
        level = "high"
        test_depth = "L1"
    elif risk_score >= 12:
        level = "medium"
        test_depth = "L2"
    else:
        level = "low"
        test_depth = "L3"

    # --- Dependencies (inherited from parent) ---
    depends_on = []
    for u, v, k, d in G.out_edges(parent_name, keys=True, data=True):
        if d.get("type") == "DEPENDS_ON":
            depends_on.append(
                {
                    "name": v,
                    "type": G.nodes[v].get("type", "?"),
                    "weight": d.get("weight", 0),
                    "via": d.get("via", "unknown"),
                }
            )
    depended_by = []
    for u, v, k, d in G.in_edges(parent_name, keys=True, data=True):
        if d.get("type") == "DEPENDS_ON" and G.nodes[u].get("type") == "Feature":
            depended_by.append(
                {
                    "name": u,
                    "weight": d.get("weight", 0),
                    "via": d.get("via", "unknown"),
                }
            )
    depends_on.sort(key=lambda x: x["weight"], reverse=True)
    depended_by.sort(key=lambda x: x["weight"], reverse=True)

    # --- Shared defect patterns (inherited) ---
    shared_patterns = []
    for u, v, k, d in G.out_edges(parent_name, keys=True, data=True):
        if d.get("type") == "SHARES_DEFECT_PATTERN":
            shared_patterns.append(
                {
                    "feature": v,
                    "shared_defect_count": d.get("shared_defect_count", 0),
                    "shared_failure_modes": d.get("shared_failure_modes", []),
                    "weight": d.get("weight", 0),
                }
            )
    shared_patterns.sort(key=lambda x: x["shared_defect_count"], reverse=True)

    # --- Recommendations ---
    recs = []
    if has_silent and len(scoped_tests) == 0:
        recs.append(
            {
                "level": "high",
                "icon": "⚠️",
                "message": f"{sum(1 for d in scoped_defects if d.get('is_silent'))} silent defects and no scoped tests — add output precision verification",
            }
        )
    elif has_silent:
        recs.append(
            {
                "level": "medium",
                "icon": "⚠️",
                "message": f"{sum(1 for d in scoped_defects if d.get('is_silent'))} silent defects in scoped view — ensure tests have reference oracle",
            }
        )
    if len(scoped_defects) >= 3:
        recs.append(
            {
                "level": "high",
                "icon": "🟠",
                "message": f"{len(scoped_defects)} scoped defects — verify regression coverage for this specific sub-area",
            }
        )
    if best_quality == 0 and len(scoped_tests) == 0:
        recs.append(
            {
                "level": "critical",
                "icon": "🔴",
                "message": "No tests specifically targeting this sub-feature — relies on parent feature tests",
            }
        )
    if any(s.get("platform_divergence") == "fully_npu_private" for s in scoped_sources):
        recs.append(
            {
                "level": "high",
                "icon": "🔒",
                "message": "NPU-private source files with no CUDA reference — precision drift hard to detect",
            }
        )

    display_name = keyword if keyword else "sub-feature"

    return {
        "feature": display_name,
        "category": parent_data.get("category", "unknown"),
        "risk_score": risk_score,
        "probability": probability,
        "impact": impact,
        "level": level,
        "test_depth": test_depth,
        "complexity": complexity,
        "n_tests": len(scoped_tests),
        "n_effective_tests": len(
            [t for t in scoped_tests if t.get("quality_score", 0) >= 3]
        ),
        "n_source_files": len(scoped_sources),
        "betweenness": parent_data.get("betweenness", 0.0),
        "degree": parent_data.get("degree", 0),
        "last_modified": parent_data.get("last_modified", ""),
        "risk_factors": {
            "probability": {
                "total": probability,
                "max": 25,
                "factors": {
                    "code_complexity": {
                        "score": code_complexity,
                        "rationale": f"Inherited from parent ({parent_name}) complexity",
                    },
                    "change_frequency": {
                        "score": change_freq,
                        "rationale": "Inherited from parent change frequency",
                    },
                    "historical_defects": {
                        "score": defect_score,
                        "rationale": f"Agent-classified: {len(scoped_defects)}/{parent_defects_total} parent defects match",
                    },
                    "dependency_depth": {
                        "score": dep_depth,
                        "rationale": "Inherited from parent dependency depth",
                    },
                    "test_effectiveness": {
                        "score": test_score,
                        "rationale": f"Agent-classified: {len(scoped_tests)}/{all_tests_count} parent tests, best quality={best_quality}/5",
                    },
                },
            },
            "impact": {
                "total": impact,
                "max": 20,
                "factors": {
                    "user_exposure": {
                        "score": user_exp,
                        "rationale": "Inherited from parent user exposure",
                    },
                    "failure_mode": {
                        "score": fail_mode,
                        "rationale": "Inherited from parent failure mode severity",
                    },
                },
            },
        },
        "defects": scoped_defects,
        "defect_count": len(scoped_defects),
        "has_silent_defects": has_silent,
        "tests": scoped_tests,
        "test_count": len(scoped_tests),
        "best_test_quality": best_quality,
        "avg_test_quality": round(
            sum(t.get("quality_score", 0) for t in scoped_tests)
            / max(1, len(scoped_tests)),
            1,
        ),
        "depends_on": depends_on,
        "depended_by": depended_by,
        "shared_defect_patterns": shared_patterns,
        "source_files": scoped_sources,
        "recommendations": recs,
        "scoping": {
            "parent_feature": parent_name,
            "keyword": keyword,
            "method": "agent",
            "match_reasons": ["agent_semantic_classification"],
            "parent_defect_count": all_defects_count,
            "parent_test_count": all_tests_count,
            "parent_source_count": all_sources_count,
            "scoped_defect_count": len(scoped_defects),
            "scoped_test_count": len(scoped_tests),
            "scoped_source_count": len(scoped_sources),
        },
    }


def sub_feature_report(
    G: nx.MultiDiGraph,
    keyword: str,
    profiles_data: dict = None,
) -> dict:
    """Generate a risk report for a sub-feature by scoping a parent feature with a keyword.

    When a user queries a name like "eagle3" that isn't a standalone Feature node,
    this searches all Features for the keyword in source_files / descriptions,
    then filters defects / tests / source_files to produce a scoped view.

    Returns a dict compatible with ``format_feature_report()``, plus an extra
    ``scoping`` key describing the parent and filter counts.
    """
    keyword_lower = keyword.lower()
    # Normalize spaces ↔ underscores so "speculative decoding" matches
    # "speculative_decoding" (the graph uses underscores for all identifiers).
    keyword_folded = keyword_lower.replace(" ", "_")

    def _kw_in(text: str) -> bool:
        """Return True if the keyword (in either form) appears in *text*."""
        t = text.lower()
        return keyword_lower in t or keyword_folded in t

    # --- Step 1: find candidate parent features ---
    candidates = []  # (feature_name, match_score, match_reasons)
    for node, data in G.nodes(data=True):
        if data.get("type") != "Feature":
            continue
        reasons = []
        score = 0
        name_str = str(node).lower()

        # Check feature name
        if _kw_in(name_str):
            reasons.append("name")
            score += 30 if name_str == keyword_folded else 15

        # Check source files
        sf_matches = 0
        for u, v, k, d in G.out_edges(node, keys=True, data=True):
            if d.get("type") == "IMPLEMENTS_IN":
                if _kw_in(str(v)):
                    sf_matches += 1
        if sf_matches > 0:
            reasons.append(f"source_files({sf_matches})")
            score += min(sf_matches * 5, 25)

        # Check description
        if _kw_in(data.get("description", "")):
            reasons.append("description")
            score += 10

        # Check models_using
        models = data.get("models_using", [])
        if any(_kw_in(str(m)) for m in models):
            reasons.append("models_using")
            score += 3

        if score > 0:
            candidates.append((node, score, reasons))

    # --- Fallback: stem matching (parent discovery only) ---
    if not candidates:
        import re as _re

        keyword_stem = _re.sub(r"\d+$", "", keyword_lower).rstrip("_-")
        if len(keyword_stem) >= 3 and keyword_stem != keyword_lower:
            for node, data in G.nodes(data=True):
                if data.get("type") != "Feature":
                    continue
                score = 0
                reasons = []
                if keyword_stem in str(node).lower():
                    reasons.append("name_stem")
                    score += 15
                sf_matches = 0
                for _, v, _k2, d in G.out_edges(node, keys=True, data=True):
                    if d.get("type") == "IMPLEMENTS_IN":
                        if keyword_stem in str(v).lower():
                            sf_matches += 1
                if sf_matches > 0:
                    reasons.append(f"source_files_stem({sf_matches})")
                    score += min(sf_matches * 5, 25)
                if keyword_stem in data.get("description", "").lower():
                    reasons.append("description_stem")
                    score += 10
                if score > 0:
                    candidates.append((node, score, reasons))

    if not candidates:
        return {
            "error": f"No feature found matching keyword '{keyword}'.",
            "hint": "Try a broader keyword, or check available features with --report.",
        }

    candidates.sort(key=lambda x: x[1], reverse=True)
    parent_name, parent_score, match_reasons = candidates[0]

    if (
        len(candidates) > 1
        and candidates[0][1] > 0
        and candidates[1][1] == candidates[0][1]
    ):
        # Tie: the second candidate has the same score — list both
        tied = [c[0] for c in candidates if c[1] == candidates[0][1]]
        return {
            "error": f"Ambiguous keyword '{keyword}': matches multiple features: {tied}.",
            "hint": "Use a more specific keyword to disambiguate.",
        }

    # --- Step 2: scope data from parent ---
    parent_data = G.nodes[parent_name]
    profiles = profiles_data or {}
    profile = profiles.get(parent_name, {})

    # Scope defects: filter by keyword/stem in defect node fields
    all_defects = []
    for u, v, k, d in G.in_edges(parent_name, keys=True, data=True):
        if d.get("type") == "AFFECTED_BY" and G.nodes[u].get("type") == "Defect":
            all_defects.append((u, G.nodes[u]))

    scoped_defects = []
    for bug_id, dn in all_defects:
        search_text = " ".join(
            [
                str(dn.get("title", "")),
                str(dn.get("root_cause", "")),
                str(dn.get("pr_title", "")),
                str(dn.get("pr_body", "")),
            ]
        )
        if _kw_in(search_text):
            scoped_defects.append((bug_id, dn))

    scoped_defects.sort(key=lambda x: str(x[1].get("date_fixed", "")), reverse=True)

    # Scope tests: filter by keyword in test path
    all_tests = []
    for u, v, k, d in G.out_edges(parent_name, keys=True, data=True):
        if d.get("type") == "TESTED_BY" and G.nodes[v].get("type") == "TestCase":
            all_tests.append((v, G.nodes[v]))

    scoped_tests = []
    for test_path, tn in all_tests:
        if _kw_in(str(test_path)):
            scoped_tests.append((test_path, tn))

    scoped_tests.sort(key=lambda x: x[1].get("quality_score", 0), reverse=True)

    # Scope source files: filter by keyword in path
    all_sources = []
    for u, v, k, d in G.out_edges(parent_name, keys=True, data=True):
        if d.get("type") == "IMPLEMENTS_IN" and G.nodes[v].get("type") == "SourceFile":
            all_sources.append((v, G.nodes[v]))

    scoped_sources = []
    for sf_path, sn in all_sources:
        if _kw_in(str(sf_path)):
            scoped_sources.append((sf_path, sn))

    scoped_sources.sort(key=lambda x: x[1].get("n_bugs", 0), reverse=True)

    has_silent = any(dn.get("is_silent", False) for _, dn in scoped_defects)

    # --- Step 3: recalculate risk factors ---
    # Complexity: inherit from parent (scoped code is a subset of same complex code)
    complexity = parent_data.get("complexity", 1)

    # Historical defects: scoped count (but score reflects parent's count as floor of severity)
    n_scoped_defects = len(scoped_defects)
    n_parent_defects = len(all_defects)
    # Defect score: proportional, but minimum 1 if any scoped defects exist
    if n_parent_defects == 0:
        defect_score = 0
    elif n_scoped_defects == 0:
        defect_score = 0  # No defects attributable to this sub-feature
    else:
        # Map proportionally: parent_score (1-5) * (scoped/parent), floor at 1
        parent_defect_score = (
            profile.get("probability", {})
            .get("factors", {})
            .get("historical_defects", {})
        )
        if isinstance(parent_defect_score, dict):
            parent_defect_score = parent_defect_score.get("score", 3)
        elif not isinstance(parent_defect_score, (int, float)):
            parent_defect_score = 3
        defect_score = max(
            1, round(parent_defect_score * n_scoped_defects / n_parent_defects)
        )

    # Test effectiveness: recalculated from scoped tests
    best_quality = max(
        (tn.get("quality_score", 0) for _, tn in scoped_tests), default=0
    )
    # test_effectiveness: higher = worse (1=excellent, 5=no tests)
    if not scoped_tests:
        test_score = 5
    elif best_quality >= 4:
        has_oracle = any(tn.get("has_oracle", False) for _, tn in scoped_tests)
        test_score = 2 if has_oracle else 3  # quality 4 but no oracle = still risky
    elif best_quality >= 3:
        test_score = 3
    elif best_quality >= 2:
        test_score = 4
    else:
        test_score = 5

    # Inherit other probability factors from parent
    prob_factors = profile.get("probability", {}).get("factors", {})
    change_freq = (
        prob_factors.get("change_frequency", {}).get("score", 1)
        if isinstance(prob_factors.get("change_frequency"), dict)
        else 1
    )
    dep_depth = (
        prob_factors.get("dependency_depth", {}).get("score", 1)
        if isinstance(prob_factors.get("dependency_depth"), dict)
        else 1
    )
    code_complexity = complexity  # from node data

    raw_prob = code_complexity + change_freq + defect_score + dep_depth + test_score
    probability = min(raw_prob, 25)

    # Inherit impact factors from parent
    imp_factors = profile.get("impact", {}).get("factors", {})
    user_exp = (
        imp_factors.get("user_exposure", {}).get("score", 1)
        if isinstance(imp_factors.get("user_exposure"), dict)
        else 1
    )
    fail_mode = (
        imp_factors.get("failure_mode", {}).get("score", 3)
        if isinstance(imp_factors.get("failure_mode"), dict)
        else 3
    )
    impact = min(user_exp + fail_mode, 20)
    risk_score = probability + impact

    # Level assignment
    if risk_score >= 28:
        level = "critical"
        test_depth = "L0"
    elif risk_score >= 20:
        level = "high"
        test_depth = "L1"
    elif risk_score >= 12:
        level = "medium"
        test_depth = "L2"
    else:
        level = "low"
        test_depth = "L3"

    # --- Build report dict (same shape as feature_risk_report) ---
    defects_out = [
        {
            "bug_id": bug_id,
            "title": dn.get("title", ""),
            "root_cause": dn.get("root_cause", ""),
            "category": dn.get("category", "unknown"),
            "severity": dn.get("severity", 3),
            "is_silent": dn.get("is_silent", False),
            "failure_mode": dn.get("failure_mode", "unknown"),
            "date_fixed": dn.get("date_fixed", ""),
            "confidence": dn.get("confidence", 0.5),
        }
        for bug_id, dn in scoped_defects
    ]

    tests_out = [
        {
            "test_file": test_path,
            "test_class": tn.get("test_class", ""),
            "test_type": tn.get("test_type", "unknown"),
            "quality_score": tn.get("quality_score", 0),
            "has_oracle": tn.get("has_oracle", False),
            "assertion_type": tn.get("assertion_type", "unknown"),
            "n_methods": tn.get("n_methods", 0),
            "ci_suite": tn.get("ci_suite", "unknown"),
            "est_time_seconds": tn.get("est_time_seconds", 0),
        }
        for test_path, tn in scoped_tests
    ]

    # Dependencies: inherit from parent
    depends_on = []
    depended_by = []
    for u, v, k, d in G.out_edges(parent_name, keys=True, data=True):
        if d.get("type") == "DEPENDS_ON":
            depends_on.append(
                {
                    "name": v,
                    "type": G.nodes[v].get("type", "?"),
                    "weight": d.get("weight", 0),
                    "via": d.get("via", "unknown"),
                }
            )
    for u, v, k, d in G.in_edges(parent_name, keys=True, data=True):
        if d.get("type") == "DEPENDS_ON" and G.nodes[u].get("type") == "Feature":
            depended_by.append(
                {
                    "name": u,
                    "weight": d.get("weight", 0),
                    "via": d.get("via", "unknown"),
                }
            )
    depends_on.sort(key=lambda x: x["weight"], reverse=True)
    depended_by.sort(key=lambda x: x["weight"], reverse=True)

    # Shared defect patterns: inherit from parent
    shared_patterns = []
    for u, v, k, d in G.out_edges(parent_name, keys=True, data=True):
        if d.get("type") == "SHARES_DEFECT_PATTERN":
            shared_patterns.append(
                {
                    "feature": v,
                    "shared_defect_count": d.get("shared_defect_count", 0),
                    "shared_failure_modes": d.get("shared_failure_modes", []),
                    "weight": d.get("weight", 0),
                }
            )
    shared_patterns.sort(key=lambda x: x["shared_defect_count"], reverse=True)

    source_files_out = [
        {
            "path": sf_path,
            "file_name": sn.get("file_name", ""),
            "file_type": sn.get("file_type", "unknown"),
            "lines_of_code": sn.get("lines_of_code", 0),
            "n_bugs": sn.get("n_bugs", 0),
            "platform_divergence": sn.get("platform_divergence", ""),
        }
        for sf_path, sn in scoped_sources
    ]

    # Recommendations for scoped view
    recs = []
    if has_silent and len(scoped_tests) == 0:
        recs.append(
            {
                "level": "high",
                "icon": "⚠️",
                "message": f"{sum(1 for d in defects_out if d.get('is_silent'))} silent defects and no scoped tests — add output precision verification",
            }
        )
    elif has_silent:
        recs.append(
            {
                "level": "medium",
                "icon": "⚠️",
                "message": f"{sum(1 for d in defects_out if d.get('is_silent'))} silent defects in scoped view — ensure tests have reference oracle",
            }
        )
    if best_quality == 0 and len(scoped_tests) == 0:
        recs.append(
            {
                "level": "critical",
                "icon": "🔴",
                "message": "No tests specifically targeting this sub-feature — it relies on parent feature tests for coverage",
            }
        )
    if n_scoped_defects >= 3:
        recs.append(
            {
                "level": "high",
                "icon": "🟠",
                "message": f"{n_scoped_defects} scoped defects ({n_parent_defects} total in parent) — verify regression coverage for this specific sub-area",
            }
        )

    return {
        "feature": f"{keyword}",
        "category": parent_data.get("category", "unknown"),
        "risk_score": risk_score,
        "probability": probability,
        "impact": impact,
        "level": level,
        "test_depth": test_depth,
        "complexity": complexity,
        "n_tests": len(scoped_tests),
        "n_effective_tests": len(
            [tn for _, tn in scoped_tests if tn.get("quality_score", 0) >= 3]
        ),
        "n_source_files": len(scoped_sources),
        "betweenness": parent_data.get("betweenness", 0.0),
        "degree": parent_data.get("degree", 0),
        "last_modified": parent_data.get("last_modified", ""),
        "risk_factors": {
            "probability": {
                "total": probability,
                "max": 25,
                "factors": {
                    "code_complexity": {
                        "score": code_complexity,
                        "rationale": f"Inherited from parent ({parent_name}) complexity",
                    },
                    "change_frequency": {
                        "score": change_freq,
                        "rationale": "Inherited from parent change frequency",
                    },
                    "historical_defects": {
                        "score": defect_score,
                        "rationale": f"Scoped: {n_scoped_defects}/{n_parent_defects} parent defects match keyword",
                    },
                    "dependency_depth": {
                        "score": dep_depth,
                        "rationale": "Inherited from parent dependency depth",
                    },
                    "test_effectiveness": {
                        "score": test_score,
                        "rationale": f"Scoped: {len(scoped_tests)}/{len(all_tests)} parent tests, best quality={best_quality}/5",
                    },
                },
            },
            "impact": {
                "total": impact,
                "max": 20,
                "factors": {
                    "user_exposure": {
                        "score": user_exp,
                        "rationale": "Inherited from parent user exposure",
                    },
                    "failure_mode": {
                        "score": fail_mode,
                        "rationale": "Inherited from parent failure mode severity",
                    },
                },
            },
        },
        "defects": defects_out,
        "defect_count": len(defects_out),
        "has_silent_defects": has_silent,
        "tests": tests_out,
        "test_count": len(tests_out),
        "best_test_quality": best_quality,
        "avg_test_quality": round(
            sum(t.get("quality_score", 0) for _, t in scoped_tests)
            / max(1, len(scoped_tests)),
            1,
        ),
        "depends_on": depends_on,
        "depended_by": depended_by,
        "shared_defect_patterns": shared_patterns,
        "source_files": source_files_out,
        "recommendations": recs,
        # Extra scoping metadata
        "scoping": {
            "parent_feature": parent_name,
            "keyword": keyword,
            "match_reasons": match_reasons,
            "parent_defect_count": n_parent_defects,
            "parent_test_count": len(all_tests),
            "parent_source_count": len(all_sources),
            "scoped_defect_count": n_scoped_defects,
            "scoped_test_count": len(scoped_tests),
            "scoped_source_count": len(scoped_sources),
        },
    }


def format_feature_report(report: dict) -> str:
    """Render a feature risk report dict as Markdown."""
    if "error" in report:
        msg = f"**Error**: {report['error']}"
        if "hint" in report:
            msg += f"\n\n*{report['hint']}*"
        return msg

    lines = []
    scoping = report.get("scoping", None)
    # --- Header ---
    level = report["level"].upper()
    if scoping:
        parent = scoping["parent_feature"]
        kw = scoping["keyword"]
        lines.append(
            f"## {parent} ⊃ **{kw}**  [{level} | score={report['risk_score']}/45 | {report['test_depth']}]"
        )
        lines.append(f"")
        lines.append(
            f"**Scoped view** from `{parent}` · keyword: `{kw}` · "
            f"matched by: {', '.join(scoping['match_reasons'])}"
        )
        lines.append(f"")
    else:
        lines.append(
            f"## {report['feature']}  [{level} | score={report['risk_score']}/45 | {report['test_depth']}]"
        )
        lines.append(f"")
    lines.append(
        f"**Category**: {report['category']}  |  "
        f"**Complexity**: {report['complexity']}/5  |  "
        f"**Tests**: {report['test_count']} ({report['n_effective_tests']} effective)  |  "
        f"**Defects**: {report['defect_count']}"
    )
    lines.append(f"")

    # --- Risk Factors ---
    lines.append(f"### Risk Factor Breakdown")
    lines.append(f"")
    lines.append(f"| Factor | Score | Rationale |")
    lines.append(f"|--------|-------|-----------|")
    for name, fdata in report["risk_factors"]["probability"]["factors"].items():
        score = fdata.get("score", "?") if isinstance(fdata, dict) else fdata
        rationale = fdata.get("rationale", "") if isinstance(fdata, dict) else ""
        lines.append(f"| {name:25s} | {str(score):5s} | {rationale} |")
    for name, fdata in report["risk_factors"]["impact"]["factors"].items():
        score = fdata.get("score", "?") if isinstance(fdata, dict) else fdata
        rationale = fdata.get("rationale", "") if isinstance(fdata, dict) else ""
        lines.append(f"| {name:25s} | {str(score):5s} | {rationale} |")
    lines.append(f"")
    lines.append(
        f"**Probability**: {report['risk_factors']['probability']['total']}/25  |  "
        f"**Impact**: {report['risk_factors']['impact']['total']}/20  |  "
        f"**Composite**: {report['risk_score']}/45"
    )
    lines.append(f"")

    # --- Defects ---
    if scoping:
        lines.append(
            f"### Defect History ({scoping['scoped_defect_count']} scoped from {scoping['parent_defect_count']} total)"
        )
    else:
        lines.append(f"### Defect History ({report['defect_count']})")
    lines.append(f"")
    if report["defects"]:
        lines.append(f"| Bug ID | Category | Severity | Silent | Date | Description |")
        lines.append(f"|--------|----------|----------|--------|------|-------------|")
        for d in report["defects"]:
            silent_mark = "✓" if d["is_silent"] else ""
            date_short = d["date_fixed"][:10] if d["date_fixed"] else "?"
            desc = d.get("title") or d.get("root_cause", "") or ""
            if len(desc) > 100:
                desc = desc[:97] + "..."
            lines.append(
                f"| {d['bug_id']} | {d['category']:15s} | {d['severity']} | {silent_mark:6s} | {date_short} | {desc} |"
            )
    else:
        lines.append(f"_No defects recorded_")
    lines.append(f"")

    # --- Tests ---
    if scoping:
        lines.append(
            f"### Test Coverage ({scoping['scoped_test_count']} scoped from {scoping['parent_test_count']} total, "
            f"best={report['best_test_quality']}/5, avg={report['avg_test_quality']}/5)"
        )
    else:
        lines.append(
            f"### Test Coverage ({report['test_count']} tests, "
            f"best={report['best_test_quality']}/5, avg={report['avg_test_quality']}/5)"
        )
    lines.append(f"")
    if report["tests"]:
        lines.append(f"| Test File | Quality | Oracle | Type | Suite |")
        lines.append(f"|-----------|---------|--------|------|-------|")
        for t in report["tests"]:
            oracle = "✓" if t["has_oracle"] else ""
            lines.append(
                f"| {t['test_file']} | {t['quality_score']}/5 | {oracle:6s} | {t['test_type']:15s} | {t['ci_suite']} |"
            )
    else:
        lines.append(f"_No tests — **critical gap**_")
    lines.append(f"")

    # --- Dependencies ---
    if scoping:
        lines.append(f"### Dependencies (inherited from `{scoping['parent_feature']}`)")
    else:
        lines.append(f"### Dependencies")
    lines.append(f"")
    if report["depends_on"]:
        ext_deps = [d for d in report["depends_on"] if d["type"] == "Dependency"]
        feat_deps = [d for d in report["depends_on"] if d["type"] == "Feature"]
        if feat_deps:
            lines.append(
                f"**Depends on ({len(feat_deps)} features)**: "
                f"{', '.join(d['name'] for d in feat_deps)}"
            )
        if ext_deps:
            ext_str = ", ".join(
                d["name"] + " (risk=" + str(d["weight"]) + ")" for d in ext_deps
            )
            lines.append(f"**External dependencies ({len(ext_deps)})**: {ext_str}")
    else:
        lines.append(f"**Depends on**: _none_")
    if report["depended_by"]:
        lines.append(
            f"**Depended on by ({len(report['depended_by'])} features)**: "
            f"{', '.join(d['name'] for d in report['depended_by'][:8])}"
            f"{'...' if len(report['depended_by']) > 8 else ''}"
        )
    else:
        lines.append(f"**Depended on by**: _none_")
    lines.append(f"")

    # --- Shared Defect Patterns ---
    if scoping:
        lines.append(
            f"### Shared Defect Patterns ({len(report['shared_defect_patterns'])} — inherited from `{scoping['parent_feature']}`)"
        )
    else:
        lines.append(
            f"### Shared Defect Patterns ({len(report['shared_defect_patterns'])})"
        )
    lines.append(f"")
    if report["shared_defect_patterns"]:
        lines.append(f"| Feature | Shared Defects | Failure Modes |")
        lines.append(f"|---------|---------------|---------------|")
        for s in report["shared_defect_patterns"]:
            modes = ", ".join(s["shared_failure_modes"][:3])
            lines.append(
                f"| {s['feature']:35s} | {s['shared_defect_count']} | {modes} |"
            )
    else:
        lines.append(f"_No shared defect patterns_")
    lines.append(f"")

    # --- Source Files ---
    if scoping:
        lines.append(
            f"### Source Files ({scoping['scoped_source_count']} scoped from {scoping['parent_source_count']} total)"
        )
    else:
        lines.append(f"### Source Files ({len(report['source_files'])})")
    lines.append(f"")
    if report["source_files"]:
        lines.append(f"| File | Lines | Bugs | Platform |")
        lines.append(f"|------|-------|------|----------|")
        for sf in report["source_files"]:
            lines.append(
                f"| {sf['path']:60s} | {sf['lines_of_code']:5d} | {sf['n_bugs']:4d} | {sf['platform_divergence']} |"
            )
    lines.append(f"")

    # --- Recommendations ---
    lines.append(f"### Recommendations")
    lines.append(f"")
    if report["recommendations"]:
        for r in report["recommendations"]:
            lines.append(f"{r['icon']} {r['message']}")
    else:
        lines.append(f"✅ No specific recommendations — feature is well-covered.")
    lines.append(f"")

    return "\n".join(lines)


# ============================================================
# Report Generator
# ============================================================


def generate_risk_report(G: nx.MultiDiGraph = None) -> str:
    """Generate a comprehensive risk report from the graph."""
    if G is None:
        G = load_graph()

    lines = []
    lines.append("# NPU Risk Knowledge Graph — Query Report")
    lines.append(f"")
    lines.append(
        f"**Nodes**: {G.number_of_nodes()}  |  **Edges**: {G.number_of_edges()}"
    )
    lines.append("")

    # Blind spots
    lines.append("## [CRITICAL] Blind Spots (High Betweenness, Low Risk)")
    blinds, non_feature = blind_spot_detection(G)
    for b in blinds[:5]:
        lines.append(
            f"- **{b['node']}** ({b['type']}): betweenness={b['betweenness']:.3f}, "
            f"risk={b['risk_score']}, gap={b['gap']:.1f}"
        )

    # Perfect storm
    lines.append("")
    lines.append("## [HIGH] Perfect Storm (Multi-Risk Overlap)")
    storms = perfect_storm(G)
    for s in storms[:5]:
        lines.append(
            f"- **{s['feature']}**: score={s['score']}/5, risk={s['risk_score']}, "
            f"silent_defects={s['has_silent_defects']}, best_test={s['best_test_quality']}/5"
        )

    # Defect hotspots
    lines.append("")
    lines.append("## [MEDIUM] Defect Hotspots (Repeat Offenders)")
    hotspots = defect_hotspots(G)
    for h in hotspots[:5]:
        lines.append(
            f"- **{h['file']}**: {h['total_defects']} defects, "
            f"{h['silent_defects']} silent ({h['silent_ratio']:.0%})"
        )

    # Test priority
    lines.append("")
    lines.append("## [INFO] Test Investment Priority")
    priorities = test_priority_ranking(G)
    for p in priorities[:5]:
        lines.append(
            f"- **{p['feature']}**: priority_score={p['priority_score']}, "
            f"risk={p['risk_score']}, gap={p['gap']}"
        )

    # Orphan tests
    lines.append("")
    orphans = orphan_tests(G)
    if orphans:
        lines.append("## [WARNING] Orphan Tests (No Feature Association)")
        for o in orphans:
            lines.append(f"- {o}")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NPU Risk Graph Queries")
    parser.add_argument("--graph", default=None, help="Path to graph JSON")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--blind-spots", action="store_true")
    parser.add_argument("--perfect-storm", action="store_true")
    parser.add_argument("--defect-hotspots", action="store_true")
    parser.add_argument("--test-priority", action="store_true")
    parser.add_argument("--orphans", action="store_true")
    parser.add_argument(
        "--oracle-blind-spots",
        type=str,
        const="__ALL__",
        nargs="?",
        help="Show blind spots for a test (or all tests if no arg)",
    )
    parser.add_argument(
        "--recommend-oracle",
        type=str,
        metavar="FEATURE",
        help="Recommend oracle patterns for a feature based on defect profile",
    )
    parser.add_argument(
        "--pattern-coverage",
        action="store_true",
        help="Show oracle pattern usage overview",
    )
    parser.add_argument("--impact", type=str, help="File name for impact analysis")
    parser.add_argument(
        "--feature", type=str, help="Feature name for single-feature risk report"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON (for --feature)"
    )
    parser.add_argument(
        "--prepare-agent",
        type=str,
        metavar="KEYWORD",
        help="Prepare Agent classification bundle for a sub-feature keyword",
    )
    parser.add_argument(
        "--apply-agent",
        type=str,
        metavar="RESULT_JSON",
        help="Apply Agent classification result to produce scoped feature report",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="",
        help="Original query keyword (used with --apply-agent for display)",
    )
    args = parser.parse_args()

    G = load_graph(Path(args.graph) if args.graph else None)

    if args.report:
        print(generate_risk_report(G))
    elif args.blind_spots:
        blinds, _ = blind_spot_detection(G)
        for b in blinds:
            print(
                f"{b['node']:40s} {b['type']:15s} bc={b['betweenness']:.3f} risk={b['risk_score']:2d}"
            )
    elif args.perfect_storm:
        for s in perfect_storm(G):
            print(
                f"{s['feature']:35s} score={s['score']}/5 risk={s['risk_score']} "
                f"silent={s['has_silent_defects']} test_qual={s['best_test_quality']}"
            )
    elif args.defect_hotspots:
        for h in defect_hotspots(G):
            print(
                f"{h['file']:50s} {h['total_defects']} defects  "
                f"silent={h['silent_defects']}  categories={h['categories']}"
            )
    elif args.test_priority:
        for p in test_priority_ranking(G):
            print(
                f"{p['feature']:35s} priority={p['priority_score']:4d}  "
                f"risk={p['risk_score']:2d}  gap={p['gap']}"
            )
    elif args.orphans:
        for o in orphan_tests(G):
            print(f"ORPHAN: {o}")
    elif args.oracle_blind_spots is not None:
        test_id = (
            None if args.oracle_blind_spots == "__ALL__" else args.oracle_blind_spots
        )
        blinds = oracle_blind_spots(G, test_id)
        if not blinds:
            print(
                "No blind spots found — all failure modes covered by oracle patterns."
            )
        else:
            print(
                f"{'Test':50s} {'Oracle Pattern':25s} {'Blind Spot':25s} {'Silent':6s}"
            )
            print("-" * 112)
            for b in blinds:
                silent_mark = "silent" if b["is_silent"] else ""
            print(
                f"{b['test_id']:50s} {b['pattern_name']:25s} "
                f"{b['blind_spot_name']:25s} {silent_mark:6s}"
            )
    elif args.recommend_oracle:
        result = recommend_oracle_for_feature(G, args.recommend_oracle)
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Feature: {result['feature']}")
            print(f"Defect failure modes: {result['defect_failure_modes']}")
            print(f"Has silent defects: {result['has_silent_defects']}")
            print(f"Current oracle patterns: {result['current_patterns'] or '(none)'}")
            print(f"Missing patterns: {result['missing_patterns'] or '(none)'}")
            print(f"Coverage gap: {result['coverage_gap'] or '(none)'}")
            print(f"\n→ {result['recommendation']}")
    elif args.pattern_coverage:
        cov = test_pattern_coverage(G)
        print(f"Oracle patterns: {cov['n_patterns']}")
        print(
            f"Features without oracle pattern coverage: {cov['n_uncovered_features']}"
        )
        print()
        for pid, info in cov["pattern_usage"].items():
            print(
                f"  {pid:30s}  tests={info['n_tests']:2d}  "
                f"detects={info['detects']}  blind_to={[b['mode'] for b in info['blind_to']]}"
            )
        if cov["uncovered_features"]:
            print(f"\nFeatures lacking oracle pattern coverage:")
            for f in cov["uncovered_features"][:10]:
                print(
                    f"  {f['feature']:35s}  risk={f['risk_score']:2d}  tests={f['n_tests']}"
                )
    elif args.impact:
        result = impacted_features(G, args.impact)
        print(f"File: {result['file']}")
        print(f"Direct impact: {result['direct']}")
        print(f"1-hop propagation: {result['propagated_1hop']}")
        print(f"2-hop propagation: {result['propagated_2hop']}")
        print(f"Total impacted features: {result['total_impacted']}")
    elif args.prepare_agent:
        profiles_path = GRAPH_DIR.parent / "baselines" / "latest" / "risk_profiles.json"
        profiles = {}
        if profiles_path.exists():
            profiles_data = load_json(profiles_path)
            for p in profiles_data.get("profiles", []):
                profiles[p["feature"]] = p

        result = prepare_agent_bundle(G, args.prepare_agent, profiles)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            if "hint" in result:
                print(f"HINT: {result['hint']}")
            sys.exit(2)
        else:
            # Print the bundle path and workflow instruction to stdout
            print(result["workflow_instruction"])
            # Also output the bundle as JSON for piping
            print("\n--- BUNDLE_JSON ---")
            print(
                json.dumps(result["bundle"], indent=2, ensure_ascii=False, default=str)
            )
            sys.exit(0)

    elif args.apply_agent:
        profiles_path = GRAPH_DIR.parent / "baselines" / "latest" / "risk_profiles.json"
        profiles = {}
        if profiles_path.exists():
            profiles_data = load_json(profiles_path)
            for p in profiles_data.get("profiles", []):
                profiles[p["feature"]] = p

        report = apply_agent_result(G, args.apply_agent, args.keyword, profiles)
        if args.json:
            output = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        else:
            output = format_feature_report(report)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(output)

    elif args.feature:
        # Always route through Agent semantic classification — even when the
        # feature name matches exactly.  Exact match = parent is self; the Agent
        # still scopes the data to the user's actual question (test risks? defect
        # patterns? impact analysis?) instead of dumping everything.
        #
        # Load risk_profiles.json for detailed factor breakdown
        profiles_path = GRAPH_DIR.parent / "baselines" / "latest" / "risk_profiles.json"
        profiles = {}
        if profiles_path.exists():
            profiles_data = load_json(profiles_path)
            for p in profiles_data.get("profiles", []):
                profiles[p["feature"]] = p

        agent_result = prepare_agent_bundle(G, args.feature, profiles)
        if "error" not in agent_result:
            # Output Agent bundle and instruction — skill picks up and runs Agent
            print(agent_result["workflow_instruction"])
            print("\n--- BUNDLE_JSON ---")
            print(
                json.dumps(
                    agent_result["bundle"], indent=2, ensure_ascii=False, default=str
                )
            )
            sys.exit(2)  # Signal to skill: needs Agent

        # No parent feature found — exit code 2 so the skill layer can take
        # over (suggest related features, try alternate keywords, or ask user).
        # Structured output: error + hint from prepare_agent_bundle, plus the
        # full feature list so the AI has context to act on.
        print(
            json.dumps(
                {
                    "error": agent_result.get(
                        "error", f"No feature found matching '{args.feature}'"
                    ),
                    "hint": agent_result.get("hint", ""),
                    "available_features": sorted(
                        n for n, d in G.nodes(data=True) if d.get("type") == "Feature"
                    ),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        sys.exit(2)
    else:
        # Default: print summary
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        node_types = Counter(G.nodes[n].get("type", "?") for n in G.nodes())
        print(f"Node types: {dict(node_types)}")
        print("\nUse --help for query options.")
