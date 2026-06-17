#!/usr/bin/env python3
"""
Graph Delta Updater — applies delta reports to the NPU risk knowledge graph.

Design reference: npu_knowledge_graph_design.html §4.2, §6.2

Usage:
    # Apply latest delta to graph
    python .claude/skills/npu-risk-graph/scripts/update_graph.py

    # Apply specific delta
    python .claude/skills/npu-risk-graph/scripts/update_graph.py --delta abc1234567.json

    # Dry-run: validate without saving
    python .claude/skills/npu-risk-graph/scripts/update_graph.py --dry-run
"""

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx
from common import (
    BASELINE_DIR,
    DB_DIR,
    DELTA_DIR,
    GRAPH_DIR,
    ROOT,
    load_json,
    now_iso,
    save_json,
)

# ============================================================
# Snapshot management
# ============================================================

SNAPSHOT_DIR = GRAPH_DIR / "snapshots"


def _ensure_snapshot_dir():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def snapshot_graph(G: nx.MultiDiGraph) -> Path:
    """Save a timestamped copy of the graph before mutation. Returns the path."""
    _ensure_snapshot_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"{ts}_graph.json"
    data = nx.node_link_data(G, edges="edges")
    save_json(path, data)
    return path


# ============================================================
# Graph loading
# ============================================================


def load_graph(graph_path: Path = None) -> nx.MultiDiGraph:
    """Load graph from node-link JSON, normalized to MultiDiGraph."""
    if graph_path is None:
        graph_path = GRAPH_DIR / "latest_graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    data = load_json(graph_path)
    G = nx.node_link_graph(data, edges="edges")
    if not isinstance(G, nx.MultiDiGraph):
        G2 = nx.MultiDiGraph()
        G2.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            G2.add_edge(u, v, **d)
        G = G2
    return G


def save_graph(G: nx.MultiDiGraph, output_path: Path = None):
    """Save graph as node-link JSON."""
    if output_path is None:
        output_path = GRAPH_DIR / "latest_graph.json"
    data = nx.node_link_data(G, edges="edges")
    data["graph_metadata"] = {
        "name": G.graph.get("name", ""),
        "built_at": G.graph.get("built_at", ""),
        "last_updated_at": G.graph.get("last_updated_at", ""),
        "last_delta_id": G.graph.get("last_delta_id", ""),
        "git_commit": G.graph.get("git_commit", ""),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }
    # Statistics
    node_types = {}
    edge_types = {}
    for n, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    for u, v, d in G.edges(data=True):
        t = d.get("type", "unknown")
        edge_types[t] = edge_types.get(t, 0) + 1
    data["statistics"] = {"node_types": node_types, "edge_types": edge_types}
    save_json(output_path, data)
    return output_path


# ============================================================
# Metric recomputation
# ============================================================


def recompute_metrics(G: nx.MultiDiGraph):
    """Recompute degree and betweenness centrality for all nodes."""
    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)

    try:
        G_undirected = nx.Graph()
        for u, v, key in G.edges(keys=True):
            G_undirected.add_edge(u, v)
        if G_undirected.number_of_nodes() > 1:
            bc = nx.betweenness_centrality(G_undirected)
            for node in G.nodes():
                G.nodes[node]["betweenness"] = bc.get(node, 0.0)
    except Exception:
        for node in G.nodes():
            G.nodes[node]["betweenness"] = G.nodes[node].get("betweenness", 0.0)


# ============================================================
# Delta application: individual change types
# ============================================================


def _apply_risk_change(G: nx.MultiDiGraph, change: dict):
    """risk_score_change: update Feature node risk properties."""
    feat = change["feature"]
    if not G.has_node(feat):
        print(f"  [WARN] Skipping unknown feature: {feat}")
        return

    new_profile = change.get("new_profile", {})
    G.nodes[feat]["risk_score"] = change["new_score"]
    G.nodes[feat]["level"] = change["new_level"]
    G.nodes[feat]["probability"] = new_profile.get("probability", {}).get(
        "total", G.nodes[feat].get("probability", 0)
    )
    G.nodes[feat]["impact"] = new_profile.get("impact", {}).get(
        "total", G.nodes[feat].get("impact", 0)
    )
    G.nodes[feat]["test_depth"] = new_profile.get(
        "test_depth", G.nodes[feat].get("test_depth", "L3")
    )
    G.nodes[feat]["fingerprint"] = new_profile.get(
        "fingerprint", G.nodes[feat].get("fingerprint", "")
    )

    direction = change.get("direction", "unchanged")
    diff = change.get("diff", 0)
    arrow = "UP" if diff > 0 else ("DOWN" if diff < 0 else "--")
    print(
        f"  [{arrow}] {feat}: "
        f"{change['old_score']}->{change['new_score']} "
        f"({change['old_level']}->{change['new_level']}) [{direction}]"
    )


def _apply_new_feature(G: nx.MultiDiGraph, change: dict):
    """new_feature: add Feature node + IMPLEMENTS_IN edges to source files."""
    name = change["name"]
    props = change.get("props", {})
    if G.has_node(name):
        print(f"  [WARN] Feature already exists, updating: {name}")
    G.add_node(name, type="Feature", **props)

    source_files = change.get("source_files", [])
    for sf in source_files:
        sf_norm = sf.replace("\\", "/")
        if not G.has_node(sf_norm):
            G.add_node(
                sf_norm,
                type="SourceFile",
                file_name=sf_norm.split("/")[-1],
                platform_divergence=(
                    "fully_npu_private"
                    if "hardware_backend/npu" in sf_norm
                    else "partial"
                ),
            )
        G.add_edge(
            name,
            sf_norm,
            type="IMPLEMENTS_IN",
            weight=round(1.0 / max(1, len(source_files)), 2),
        )
    print(f"  + Feature: {name} ({len(source_files)} source files)")


def _apply_source_change(G: nx.MultiDiGraph, change: dict):
    """feature_source_change: update IMPLEMENTS_IN edges for a feature."""
    feat = change["feature"]
    if not G.has_node(feat):
        print(f"  [WARN] Unknown feature: {feat}")
        return
    # Remove old IMPLEMENTS_IN edges
    edges_to_remove = []
    for u, v, k, d in G.out_edges(feat, keys=True, data=True):
        if d.get("type") == "IMPLEMENTS_IN":
            edges_to_remove.append((u, v, k))
    for u, v, k in edges_to_remove:
        G.remove_edge(u, v, k)

    # Add new IMPLEMENTS_IN edges
    new_sources = change.get("source_files", [])
    for sf in new_sources:
        sf_norm = sf.replace("\\", "/")
        if not G.has_node(sf_norm):
            G.add_node(
                sf_norm,
                type="SourceFile",
                file_name=sf_norm.split("/")[-1],
                platform_divergence=(
                    "fully_npu_private"
                    if "hardware_backend/npu" in sf_norm
                    else "partial"
                ),
            )
        G.add_edge(
            feat,
            sf_norm,
            type="IMPLEMENTS_IN",
            weight=round(1.0 / max(1, len(new_sources)), 2),
        )
    print(f"  ~ Source change: {feat} -> {len(new_sources)} files")


def _apply_new_test(G: nx.MultiDiGraph, change: dict):
    """new_test: add TestCase node + TESTED_BY / COVERS edges."""
    test_id = change.get("test_id", change.get("test_file", ""))
    props = change.get("props", {})
    if G.has_node(test_id):
        print(f"  [WARN] Test already exists, updating: {test_id}")
    G.add_node(test_id, type="TestCase", **props)

    # TESTED_BY edges
    for feat_name in change.get("features_tested", []):
        if G.has_node(feat_name):
            G.add_edge(
                feat_name,
                test_id,
                type="TESTED_BY",
                weight=props.get("quality_score", 1),
            )
            # COVERS: via IMPLEMENTS_IN chain
            for u, v, k, d in G.out_edges(feat_name, keys=True, data=True):
                if d.get("type") == "IMPLEMENTS_IN":
                    G.add_edge(
                        test_id,
                        v,
                        type="COVERS",
                        weight=round(
                            d.get("weight", 0.5) * props.get("quality_score", 1) / 5, 2
                        ),
                    )
    print(f"  + Test: {test_id} ({len(change.get('features_tested', []))} features)")


def _apply_test_removed(G: nx.MultiDiGraph, change: dict):
    """test_removed: remove TestCase node and all its edges."""
    test_id = change.get("test_id", change.get("test_file", ""))
    if G.has_node(test_id):
        G.remove_node(test_id)
        print(f"  - Test removed: {test_id}")
    else:
        print(f"  [WARN] Test not found: {test_id}")


def _apply_new_defect(G: nx.MultiDiGraph, change: dict):
    """new_defect: add Defect node + AFFECTED_BY / FIXED_IN edges."""
    bug_id = change.get("bug_id", "")
    props = change.get("props", {})
    if G.has_node(bug_id):
        print(f"  [WARN] Defect already exists, updating: {bug_id}")
    G.add_node(bug_id, type="Defect", **props)

    for feat_name in change.get("features_affected", []):
        if G.has_node(feat_name):
            G.add_edge(
                bug_id, feat_name, type="AFFECTED_BY", weight=props.get("severity", 3)
            )

    for file_path in change.get("files_fixed", []):
        file_norm = file_path.replace("\\", "/")
        if G.has_node(file_norm) and G.nodes[file_norm].get("type") == "SourceFile":
            G.add_edge(
                bug_id,
                file_norm,
                type="FIXED_IN",
                weight=round(1.0 / max(1, len(change.get("files_fixed", []))), 2),
            )

    # Also update SHARES_DEFECT_PATTERN: when a new defect is added, it may
    # create new shared-defect couplings between features.
    affected = [fn for fn in change.get("features_affected", []) if G.has_node(fn)]
    if len(affected) >= 2:
        mode = props.get("failure_mode", "unknown")
        for i, a in enumerate(affected):
            for b in affected[i + 1 :]:
                # Increment existing or create new SHARES_DEFECT_PATTERN edge
                existing = False
                for u, v, k, d in G.out_edges(a, keys=True, data=True):
                    if d.get("type") == "SHARES_DEFECT_PATTERN" and v == b:
                        new_count = d.get("shared_defect_count", 1) + 1
                        modes = set(d.get("shared_failure_modes", []))
                        modes.add(mode)
                        G.edges[a, v, k].update(
                            {
                                "shared_defect_count": new_count,
                                "shared_failure_modes": sorted(modes),
                                "weight": min(5, new_count),
                            }
                        )
                        existing = True
                        break
                if not existing:
                    G.add_edge(
                        a,
                        b,
                        type="SHARES_DEFECT_PATTERN",
                        weight=1,
                        shared_defect_count=1,
                        shared_failure_modes=[mode],
                    )
                    G.add_edge(
                        b,
                        a,
                        type="SHARES_DEFECT_PATTERN",
                        weight=1,
                        shared_defect_count=1,
                        shared_failure_modes=[mode],
                    )

    print(
        f"  + Defect: {bug_id} -> {len(change.get('features_affected', []))} features, "
        f"{len(change.get('files_fixed', []))} files"
    )


# ============================================================
# Main delta application
# ============================================================

# Dispatch table for change types
_CHANGE_HANDLERS = {
    "risk_score_change": _apply_risk_change,
    "new_feature": _apply_new_feature,
    "feature_source_change": _apply_source_change,
    "new_test": _apply_new_test,
    "test_removed": _apply_test_removed,
    "new_defect": _apply_new_defect,
}


def apply_delta(G: nx.MultiDiGraph, delta_report: dict) -> nx.MultiDiGraph:
    """Apply a delta report to the graph, mutating G in place.

    Args:
        G: The knowledge graph (MultiDiGraph), mutated in place.
        delta_report: A delta record produced by run_delta.py's generate_delta_report().

    Returns:
        G (same object, mutated).
    """
    changes = delta_report.get("changes", [])
    if not changes:
        print("  -> No changes to apply.")
        return G

    applied = 0
    skipped = 0
    for change in changes:
        change_type = change.get("type", "unknown")
        handler = _CHANGE_HANDLERS.get(change_type)
        if handler:
            handler(G, change)
            applied += 1
        else:
            print(f"  [WARN] Unknown change type: {change_type}")
            skipped += 1

    print(
        f"  -> Applied {applied} changes" + (f", skipped {skipped}" if skipped else "")
    )

    # Update graph metadata
    G.graph["last_updated_at"] = delta_report.get("generated_at", "")
    G.graph["last_delta_id"] = delta_report.get("delta_id", "")
    G.graph["git_commit"] = delta_report.get(
        "current_commit", G.graph.get("git_commit", "")
    )

    # Recompute metrics
    recompute_metrics(G)

    return G


def apply_delta_from_file(
    graph_path: Path = None, delta_path: Path = None, dry_run: bool = False
) -> nx.MultiDiGraph:
    """Load graph and delta from files, apply, save.

    Args:
        graph_path: Path to graph JSON (default: latest_graph.json).
        delta_path: Path to delta JSON (default: deltas/latest.json).
        dry_run: If True, validate and print but don't save.

    Returns:
        The updated graph.
    """
    if delta_path is None:
        delta_path = DELTA_DIR / "latest.json"

    if not delta_path.exists():
        raise FileNotFoundError(
            f"Delta not found: {delta_path}\n"
            "Run `/npu-risk-graph update` first (or: python .claude/skills/npu-risk-graph/scripts/run_delta.py)."
        )

    delta_report = load_json(delta_path)
    print(f"Delta: {delta_report.get('delta_id', 'unknown')}")
    print(f"  Parent commit: {delta_report.get('parent_commit', '?')[:12]}...")
    print(f"  Current commit: {delta_report.get('current_commit', '?')[:12]}...")
    print(f"  Changes: {len(delta_report.get('changes', []))}")
    print(f"  Direct features: {len(delta_report.get('direct_features', []))}")
    print(f"  Propagated features: {len(delta_report.get('propagated_features', []))}")
    print()

    # Load graph
    G = load_graph(graph_path)
    print(f"Graph before: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Snapshot before mutation
    snap_path = snapshot_graph(G)
    print(f"Snapshot saved: {snap_path}")

    # Apply
    print(f"\nApplying delta...")
    apply_delta(G, delta_report)

    print(f"\nGraph after: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if dry_run:
        print("\n[Dry-run] Changes validated. Graph NOT saved.")
    else:
        output = save_graph(G, graph_path)
        print(f"\nGraph updated: {output}")

    return G


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply delta report to NPU risk graph")
    parser.add_argument(
        "--delta", default=None, help="Delta JSON path (default: deltas/latest.json)"
    )
    parser.add_argument(
        "--graph",
        default=None,
        help="Graph JSON path (default: graph/latest_graph.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate changes without saving"
    )
    args = parser.parse_args()

    delta_path = Path(args.delta) if args.delta else None
    graph_path = Path(args.graph) if args.graph else None

    try:
        apply_delta_from_file(
            graph_path=graph_path, delta_path=delta_path, dry_run=args.dry_run
        )
        print("\nDone.")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
